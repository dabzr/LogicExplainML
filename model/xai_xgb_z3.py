from z3 import *


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
        self.T_constraints = self.feature_constraints_expression(self.data)
        self.T_model = self.model_trees_expression(self.model)
        self.T = And(self.T_model, self.T_constraints)
        self.label_proportions = labels.mean()

    def explain(self, instance, reorder="asc"):
        self.D = self.decision_function_expression(
            self.model, [instance], self.label_proportions
        )
        self.I = self.instance_expression(instance)

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

            if len(unique_values) == 2:
                a, b = unique_values
                constraint = Or(x == RealVal(a), x == RealVal(b))
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
        init_value = label_proportions

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
            final_equation = compare_equation

        return And(final_equation)

    def instance_expression(self, instance):
        formula = [Real(self.columns[i]) == value for i, value in enumerate(instance)]
        return formula

    def explain_expression(self, I, T, D, model, reorder):
        X = I.copy()
        T_s = simplify(T)
        D_s = simplify(D)

        importances = model.feature_importances_
        non_zero_indices = np.where(importances != 0)[0]

        if reorder == "asc":
            sorted_feature_indices = non_zero_indices[
                np.argsort(importances[non_zero_indices])
            ]
            X = [X[i] for i in sorted_feature_indices]
        elif reorder == "desc":
            sorted_feature_indices = non_zero_indices[
                np.argsort(-importances[non_zero_indices])
            ]
            X = [X[i] for i in sorted_feature_indices]

        for feature in X.copy():
            X.remove(feature)

            # prove(Implies(And(And(X), T), D))
            if self.is_proved(Implies(And(And(X), T_s), D_s)):
                continue
                # print('proved')
            else:
                # print('not proved')
                X.append(feature)

        return X

    def is_proved(self, f):
        s = Solver()
        s.add(Not(f))
        if s.check() == unsat:
            return True
        else:
            return False
