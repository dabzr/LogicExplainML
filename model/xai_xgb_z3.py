from z3 import *
import numpy as np


class XGBoostExplainer:
    """Apenas classificação binária e base_score = None
    data = X. labels = y
    """

    def __init__(self, model, data, labels):
        """_summary_

        Args:
            model (XGBoost): xgboost model fited
            data (DataFrame): dataframe (X or X_train)
            labels (array): y (targets)
        """
        self.model = model
        self.data = data.values
        self.columns = data.columns
        self.categoric_features = []
        self.max_categories = 2
        self.T_constraints = self.feature_constraints_expression(self.data)
        self.T_model = self.model_trees_expression(self.model)
        self.T = And(self.T_model, self.T_constraints)
        self.label_proportions = labels.mean()

    def explain(self, instance, reorder="asc"):
        self.I = self.instance_expression(instance)
        self.D = self.decision_function_expression(
            self.model, [instance], self.label_proportions
        )

        return self.explain_expression(self.I, self.T, self.D, self.model, reorder)

    def feature_constraints_expression(self, X: np.ndarray):
        """
        Recebe um dataset e retorna uma fórmula no z3 com:
        - Restrições de valor máximo e mínimo para features contínuas.
        - Restrições de igualdade para features categóricas binárias.
        """
        constraints = []

        for i in range(X.shape[1]):
            feature_values = X[:, i]
            unique_values = np.unique(feature_values)

            x = Real(self.columns[i])
            if len(unique_values) <= self.max_categories:
                self.categoric_features.append(self.columns[i])

                constraint = []
                for unique_value in unique_values:
                    constraint.append(x == RealVal(unique_value))
                constraint = Or(constraint)
            else:
                min_val, max_val = feature_values.min(), feature_values.max()
                min_val_z3 = RealVal(min_val)
                max_val_z3 = RealVal(max_val)
                constraint = And(min_val_z3 <= x, x <= max_val_z3)

            constraints.append(constraint)

        return And(*constraints)

    def model_trees_expression(self, model):
        """
        Constrói expressões lógicas para todas as árvores de decisão em um dataframe de XGBoost.
        Para árvores que são apenas folhas, gera diretamente um And com o valor da folha.

        Args:
            df (pd.DataFrame): Dataframe contendo informações das árvores.
            class_index (int): Índice da classe atual.

        Returns:
            z3.ExprRef: Fórmula representando todos os caminhos de todas as árvores.
        """
        df = model.get_booster().trees_to_dataframe()
        df['Split'] = df['Split'].round(4)
        self.booster_df = df
        class_index = 0  # if model.n_classes_ == 2:

        all_tree_formulas = []

        for tree_index in df["Tree"].unique():
            tree_df = df[df["Tree"] == tree_index]
            o = Real(f"o_{tree_index}_{class_index}")

            if len(tree_df) == 1 and tree_df.iloc[0]["Feature"] == "Leaf":
                leaf_value = tree_df.iloc[0]["Gain"]
                all_tree_formulas.append(And(o == leaf_value))
                continue

            path_formulas = []

            def get_conditions(node_id):
                conditions = []
                current_node = tree_df[tree_df["ID"] == node_id]
                if current_node.empty:
                    return conditions

                parent_node = tree_df[
                    (tree_df["Yes"] == node_id) | (tree_df["No"] == node_id)
                ]
                if not parent_node.empty:
                    parent_data = parent_node.iloc[0]
                    feature = parent_data["Feature"]
                    split_value = parent_data["Split"]
                    x = Real(feature)
                    if parent_data["Yes"] == node_id:
                        conditions.append(x < split_value)
                    else:
                        conditions.append(x >= split_value)
                    conditions = get_conditions(parent_data["ID"]) + conditions

                return conditions

            for _, node in tree_df[tree_df["Feature"] == "Leaf"].iterrows():
                leaf_value = node["Gain"]
                leaf_id = node["ID"]
                conditions = get_conditions(leaf_id)
                path_formula = And(*conditions)
                implication = Implies(path_formula, o == leaf_value)
                path_formulas.append(implication)

            all_tree_formulas.append(And(*path_formulas))

        return And(*all_tree_formulas)

    def decision_function_expression(self, model, x, label_proportions):
        n_classes = 1 if model.n_classes_ <= 2 else model.n_classes_
        predicted_class = model.predict(x)[0]
        n_estimators = len(model.get_booster().get_dump())

        estimator_pred = Solver()
        estimator_pred.add(self.I)
        estimator_pred.add(self.T)
        variables = [Real(f'o_{i}_0') for i in range(n_estimators)]
        if estimator_pred.check() == sat:
            solvermodel = estimator_pred.model()
            total_sum = sum(float(solvermodel.eval(var).as_fraction())
                            for var in variables)
        else:
            total_sum = 0
            print('estimator error')
        init_value = model.predict(x, output_margin=True)[0] - total_sum
        # print('margin', model.predict(x, output_margin=True)[0], 'estsum', total_sum)
        # print('init', init_value)

        equation_list = []
        for class_number in range(n_classes):
            estimator_list = []
            for estimator_number in range(
                int(len(model.get_booster().get_dump()) / n_classes)
            ):
                o = Real(f"o_{estimator_number}_{class_number}")
                estimator_list.append(o)
            equation_o = Sum(estimator_list) + init_value
            equation_list.append(equation_o)

        if n_classes <= 2:
            if predicted_class == 0:
                final_equation = equation_list[0] < 0
            else:
                final_equation = equation_list[0] > 0
        else:
            compare_equation = []
            for class_number in range(n_classes):
                if predicted_class != class_number:
                    compare_equation.append(
                        equation_list[predicted_class] > equation_list[class_number]
                    )
            final_equation = And(compare_equation)

        return final_equation

    def instance_expression(self, instance):
        formula = [Real(self.columns[i]) == value for i,
                   value in enumerate(instance)]
        return formula

    def explain_expression(self, I, T, D, model, reorder):
        i_expression = I.copy()
        T_s = T
        D_s = D

        importances = model.feature_importances_
        non_zero_indices = np.where(importances != 0)[0]

        if reorder == "asc":
            sorted_feature_indices = non_zero_indices[
                np.argsort(importances[non_zero_indices])
            ]
            i_expression = [i_expression[i] for i in sorted_feature_indices]
        elif reorder == "desc":
            sorted_feature_indices = non_zero_indices[
                np.argsort(-importances[non_zero_indices])
            ]
            i_expression = [i_expression[i] for i in sorted_feature_indices]

        for feature in i_expression.copy():

            i_expression.remove(feature)

            # prove(Implies(And(And(i_expression), T), D))
            if self.is_proved(Implies(And(And(i_expression), T_s), D_s)):
                continue
                # print('proved')
            else:
                # print('not proved')
                i_expression.append(feature)
        # print(self.is_proved(Implies(And(And(i_expression), T_s), D_s)))
        return i_expression

    def is_proved(self, f):
        s = Solver()
        s.add(Not(f))
        if s.check() == unsat:
            return True
        else:
            return False

    def delta_expression(self, exp):
        expressions = []
        delta = Real('delta')

        self.delta_features = []
        for name in exp:
            tokens = name.split(" == ")
            z3feature = Real(tokens[0])
            self.delta_features.append(str(z3feature))
            value = tokens[1]

            if tokens[0] in self.categoric_features:
                expressions.append(z3feature == float(value))
            else:
                expressions.append(z3feature >= float(value) - delta)
                expressions.append(z3feature <= float(value) + delta)

        expressions.append(delta >= 0)
        self.deltaexp = expressions
        return expressions

    def explain_range(self, instance, reorder="asc"):
        exp = self.explain(instance, reorder)
        if exp != []:
            expstr = []
            for expression in exp:
                expstr.append(str(expression))
            self.delta_expressions = self.delta_expression(expstr)

            opt = Optimize()
            opt.add(self.delta_expressions)
            opt.add(self.T)
            opt.add(Not(self.D))

            delta = Real('delta')
            expmin = opt.minimize(delta)
            opt.check()

            rangemodel = opt.model()

            value = str(expmin.value())
            # print(value)

            if "+ epsilon" in value:
                delta_value = float(value.split(" + ")[0])
            elif "epsilon" == value:
                delta_value = 0
                expstr = []
                for exppart in exp:
                    expstr.append(str(exppart))
                return expstr
            else:
                delta_value = float(value) - 0.01
            range_exp = []

            # for declaration in rangemodel.decls():
            #     if declaration.name() in self.delta_features:
            #       print(f"{declaration.name()}: {rangemodel[declaration]}")
            # print(delta_value)

            for item in exp:
                name = str(item.arg(0))
                if name not in self.categoric_features:
                    idx = list(self.columns).index(name)
                    min_idx = np.min(self.data[:, idx])
                    max_idx = np.max(self.data[:, idx])


                    itemvalue = float(item.arg(1).as_fraction())

                    lower = itemvalue - delta_value
                    if lower < min_idx:
                        lower = min_idx

                    upper = itemvalue + delta_value
                    if upper > max_idx:
                        upper = max_idx

                    # print(itemvalue, lower, upper)
                    range_exp.append(f'{lower} <= {name} <= {upper}')
                else:
                    range_exp.append(f'{name} == {item.arg(1)}')

            # for item in exp:
            #     if str(item.arg(0)) not in self.categoric_features:
            #         test = opt.minimize(Real(str(item.arg(0))))
            #         opt.check()
            #         lower = float(str(test.value()))

            #         test = opt.maximize(Real(str(item.arg(0))))
            #         opt.check()
            #         upper = float(str(test.value()))
            #         # itemvalue = float(item.arg(1).as_fraction())
            #         # lower = round(itemvalue - delta_value, 6)
            #         # upper = round(itemvalue + delta_value, 2)
            #         range_exp.append(f'{lower} <= {item.arg(0)} <= {upper}')
            #     else:
            #         range_exp.append(f'{item.arg(0)} == {item.arg(1)}')

            return range_exp
        else:
            return exp