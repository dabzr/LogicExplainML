from z3 import *
import numpy as np


class Explainer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.T_constraints = self.feature_constraints_expression(self.data)
        self.T_model = self.model_trees_expression(self.model)
        self.T = And(self.T_model, self.T_constraints)

    def explain(self, instance, reorder="asc"):
        self.D = self.decision_function_expression(self.model, [instance])
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

            x = Real(f"x{i}")

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
        formulas = []
        for i, estimators in enumerate(model.estimators_):
            for class_index, estimator in enumerate(estimators):
                formula = self.tree_paths_expression(estimator, i, class_index)
                formulas.append(formula)
        return And(*formulas)

    def tree_paths_expression(self, tree, tree_index, class_index):
        tree_ = tree.tree_
        feature = tree_.feature
        threshold = tree_.threshold
        value = tree_.value

        paths = []
        o = Real(f"o_{tree_index}_{class_index}")

        def traverse(node, path_conditions):

            if feature[node] == -2:
                leaf_value = value[node][0][0]
                path_formula = And(path_conditions)
                implication = Implies(path_formula, o == leaf_value)
                paths.append(implication)
            else:

                x = Real(f"x{feature[node]}")
                left_condition = x <= threshold[node]
                right_condition = x > threshold[node]
                traverse(tree_.children_left[node], path_conditions + [left_condition])
                traverse(
                    tree_.children_right[node], path_conditions + [right_condition]
                )

        traverse(0, [])
        return And(*paths)

    def decision_function_expression(self, model, x):
        learning_rate = model.learning_rate
        estimators = model.estimators_
        n_classes = 1 if model.n_classes_ <= 2 else model.n_classes_

        decision = model.decision_function(x)
        predicted_class = model.predict(x)[0]

        estimator_results = []
        for estimator in estimators:
            class_predictions = [tree.predict(x) for tree in estimator]
            estimator_results.append(class_predictions)

        estimator_sum = np.sum(estimator_results, axis=0) * learning_rate
        init_value = decision - estimator_sum.T

        equation_list = []
        for class_number in range(n_classes):
            estimator_list = []
            for estimator_number in range(len(estimators)):
                o = Real(f"o_{estimator_number}_{class_number}")
                estimator_list.append(o)
            equation_o = (
                Sum(estimator_list) * learning_rate + init_value[0][class_number]
            )
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
        formula = [Real(f"x{i}") == value for i, value in enumerate(instance)]
        return formula

    def explain_expression(self, I, T, D, model, reorder):
        X = I.copy()
        T_s = simplify(T)
        D_s = simplify(D)

        importances = model.feature_importances_
        non_zero_indices = np.where(importances != 0)[0]

        if reorder == "asc":
            sorted_feature_indices = non_zero_indices[np.argsort(importances[non_zero_indices])]
            X = [X[i] for i in sorted_feature_indices]
        elif reorder == "desc":
            sorted_feature_indices = non_zero_indices[np.argsort(-importances[non_zero_indices])]
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
