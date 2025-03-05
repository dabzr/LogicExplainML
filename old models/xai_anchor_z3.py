from z3 import *

class ExplainerCompleter:
    def __init__(self, model, data, round=None):
        self.model = model

        # model T
        self.T_constraints = self.feature_constraints_expression(data)
        self.T_model = self.model_trees_expression(self.model)
        self.T = And(self.T_model, self.T_constraints)

    def explain_instance(self, instance, exp, verbose=False, delta_fix = True):
        opt = Optimize()
        self.exp = exp

        # anchor expressions
        anchor_expressions, anchor_features = anchor_z3_expression(exp.names())
        self.anchor_expressions = anchor_expressions
        self.anchor_features = anchor_features
        opt.add(anchor_expressions)

        # delta
        # delta >= 0
        # todas as features que não estao no anchor > fazer as igualdades delta
        anchor_variables = []
        for formula in anchor_expressions:
            anchor_variables.append(str(formula.arg(0)))

        feature_names = [f"x{i}" for i in range(instance.shape[0])]

        if delta_fix == True:
            delta = Int("delta")
        else:
            delta = Real("delta")
        opt.add(delta >= 0)
        for i, var in enumerate(feature_names):
            if var not in anchor_variables:  # and importance_dic[var] != 0:
                z3_var = Real(var)
                opt.add(
                    (instance[i]) - delta <= z3_var, z3_var <= (instance[i]) + delta
                )
                # print(f'{instance[i]} - {delta} <= {var}, {var} <= {instance[i]} + {delta}')

        # not D
        self.D = decision_function_expression(self.model, [instance])

        # model
        opt.add(self.T)
        opt.add(Not(self.D))

        # minimize delta
        opt.minimize(delta)
        if opt.check() == sat:
            if verbose:
                for var in opt.model():
                    print(var, "=", opt.model()[var])
            print("delta =", opt.model().eval(delta))
        else:
            print("(unsat == correct)")

    def feature_constraints_expression(X, round = 0):
        constraints = []

        for i in range(X.shape[1]):
            feature_values = X[:, i] * 10**round
            # np.unique
            min_val, max_val = feature_values.min(), feature_values.max()

            x = Real(f"x{i}")
            min = RealVal(min_val)
            max = RealVal(max_val)

            constraint = And(min <= x, x <= max)
            constraints.append(constraint)

        return And(*constraints)
    
    def model_trees_expression(self, model):
        formulas = []
        for i, estimators in enumerate(model.estimators_):
            for class_index, estimator in enumerate(estimators):
                formula = self.tree_paths_expression(estimator, i, class_index)
                formulas.append(formula)
        return And(*formulas)

    def tree_paths_expression(tree, tree_index, class_index, round = 0):
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
                traverse(tree_.children_right[node], path_conditions + [right_condition])

        traverse(0, [])
        return And(*paths)
    
    def make_expression(feature, operator, value):
        z3feature = Real(feature)
        if operator == "<=":
            expression = z3feature <= float(value)
        elif operator == ">=":
            expression = z3feature >= float(value)
        elif operator == "<":
            expression = z3feature < float(value)
        elif operator == ">":
            expression = z3feature > float(value)
        elif operator == "==" or operator == "=":
            expression = z3feature == float(value)
        return expression


    def anchor_z3_expression(exp):
        pattern = r"x\d+"
        operator_map = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "=": "=", "==": "=="}

        expressions = []
        features = []
        for name in exp:
            tokens = name.split(" ")
            match = re.search(pattern, name)

            if match:
                feature = match.group()
                if tokens[0] == feature:
                    operator, value = tokens[1], tokens[2]
                    expressions.append(make_expression(feature, operator, value))
                elif tokens[2] == feature and len(tokens) < 5:
                    operator = operator_map[tokens[1]]
                    value = tokens[0]
                    expressions.append(make_expression(feature, operator, value))
                elif len(tokens) == 5:
                    operator1 = operator_map[tokens[1]]
                    operator2 = tokens[3]
                    expressions.append(make_expression(feature, operator1, tokens[0]))
                    expressions.append(make_expression(feature, operator2, tokens[4]))
                else:
                    print("expression error")
                    continue
                features.append(feature)

        return expressions, features