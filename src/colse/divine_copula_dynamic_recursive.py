import numpy as np
from colse.copula_functions import get_copula, gumbel_grad
from colse.copula_types import CopulaTypes
from loguru import logger


class DivineCopulaDynamicRecursive:
    EPSILON = 1e-3
    GRAD_ENABLED = False

    def __init__(self, theta_dict=None, copula_type_dict=None, copula_type = CopulaTypes.GUMBEL, verbose=False) -> None:
        self.show_p_values = False
        self.verbose = verbose
        self.copula_type = copula_type
        self.theta_dict = (
            {(i, j): 1.5 for i in range(0, 10) for j in range(0, 10) if i != j} if theta_dict is None else theta_dict
        )
        self.copula_type_dict = copula_type_dict
        self.copula_cache = {}
        self.exec_count = 0

    def get_cahed_copula(self, *args):
        key_name = self.get_p_key(*args)
        if key_name in self.copula_cache:
            # print(f"Cache Hit : {key_name}")
            return self.copula_cache[key_name]
        return None
    
    def store_copula(self, *args, value):
        key_name = self.get_p_key(*args)
        self.copula_cache[key_name] = value

    @staticmethod
    def get_key(name_1, name_2):
        column_id_1 = int(name_1.split("_")[0])
        column_id_2 = int(name_2.split("_")[0])
        col_list = [column_id_1 - 1, column_id_2 - 1]
        col_list.sort()
        return tuple(col_list)

    def get_theta_value(self, name_1, name_2):
        return self.theta_dict[self.get_key(name_1, name_2)]
    
    def get_copula_type(self, name_1, name_2):
        copula_type = self.copula_type
        if self.copula_type_dict is not None:
            copula_type = self.copula_type_dict[self.get_key(name_1, name_2)]
        return copula_type

    def get_p_key(self, *args):
        #  TODO - 1,2, 10 example
        value_list = []
        for arg in args:
            if isinstance(arg, list):
                value_list.extend(arg.copy())
            else:
                value_list.append(arg)

        value_list = [str(v) for v in value_list]
        value_list.sort()
        p_key = ",".join(value_list)
        print(f"P-Values : {p_key}") if self.show_p_values else None
        return p_key

    def predict(self, lb_ub_list, column_list=None):
        self.copula_cache = {}
        self.exec_count = 0

        if len(lb_ub_list) == 0:
            self.exec_count += 1
            return 1
        if len(lb_ub_list) == 2:
            self.exec_count += 1
            ce = lb_ub_list[1] - lb_ub_list[0]
            return ce if ce > 0 else self.EPSILON
        
        # This might not happen at all, but just in case | TODO: Check performance impact and remove if needed
        # is_any_two_values_equal = np.any(lb_ub_list[0::2] == lb_ub_list[1::2])
        # if is_any_two_values_equal:
        #     logger.warning(f"Any two values are equal : {lb_ub_list}")
        #     return 0

        p_values = {}

        no_of_columns = len(lb_ub_list) // 2
        if column_list is None:
            column_list = [i + 1 for i in range(no_of_columns)]
        else:
            assert isinstance(column_list, list), "Column List should be a list"
            assert column_list[0] > 0, "Column Index should start from 1"
            assert len(column_list) == no_of_columns, "Column List should be of same length as lb_ub_list"

        logger.info(f"Column List : {column_list}") if self.verbose else None

        if len(lb_ub_list) == 4:
            theta = self.get_theta_value(f"{column_list[0]}_T", f"{column_list[1]}_T")
            copula_type = self.get_copula_type(f"{column_list[0]}_T", f"{column_list[1]}_T")
            lb_lb = get_copula(copula_type, cdf1=lb_ub_list[0], cdf2=lb_ub_list[2], theta=theta)
            lb_ub = get_copula(copula_type, cdf1=lb_ub_list[0], cdf2=lb_ub_list[3], theta=theta)
            ub_lb = get_copula(copula_type, cdf1=lb_ub_list[1], cdf2=lb_ub_list[2], theta=theta)
            ub_ub = get_copula(copula_type, cdf1=lb_ub_list[1], cdf2=lb_ub_list[3], theta=theta)
            self.exec_count += 4
            ce = lb_lb + ub_ub - lb_ub - ub_lb
            if ce <= 0:
                if lb_ub_list[0] == lb_ub_list[1] and lb_ub_list[2] == lb_ub_list[3]:
                    return self.EPSILON
                elif lb_ub_list[0] == lb_ub_list[1]:
                    return (lb_ub_list[3] - lb_ub_list[1]) * self.EPSILON
                elif lb_ub_list[2] == lb_ub_list[3]:
                    return (lb_ub_list[2] - lb_ub_list[0]) * self.EPSILON
                else:
                    return self.EPSILON
            return ce

        it_left_bound = f"{column_list[0]}"
        it_middle_bound = [f"{i}" for i in column_list[1:-1]]
        it_right_bound = f"{column_list[-1]}"

        def get_cdf_value(name):
            column_id, bound = name.split("_")
            column_id = int(column_id)
            column_index = column_list.index(column_id)
            add = 1 if bound == "ub" else 0
            return lb_ub_list[add + column_index * 2]

        def compute(left_bound, middle_bound, right_bound, level=0):
            (
                print(f"Going inside... Level {level} --------- {left_bound},{right_bound} | {middle_bound}")
                if self.verbose
                else None
            )
            if len(middle_bound) == 1:
                
                # print("\n", "=" * 20, "Leaf start") if self.verbose else None
                middle_value = middle_bound[0]
                middle_lb = f"{middle_value}_lb"
                middle_ub = f"{middle_value}_ub"

                copula_ll = self.get_cahed_copula(left_bound, middle_lb)
                if copula_ll is None:
                    self.exec_count += 1
                    copula_ll = get_copula(
                        copula_type=self.get_copula_type(left_bound, middle_lb),
                        cdf1=get_cdf_value(left_bound),
                        cdf2=get_cdf_value(middle_lb),
                        theta=self.get_theta_value(left_bound, middle_lb),
                    )
                    self.store_copula(left_bound, middle_lb, value=copula_ll)

                copula_lr = self.get_cahed_copula(left_bound, middle_ub)
                if copula_lr is None:
                    self.exec_count += 1
                    copula_lr = get_copula(
                        copula_type=self.get_copula_type(left_bound, middle_ub),
                        cdf1=get_cdf_value(left_bound),
                        cdf2=get_cdf_value(middle_ub),
                        theta=self.get_theta_value(left_bound, middle_ub),
                    )
                    self.store_copula(left_bound, middle_ub, value=copula_lr)

                copula_rl = self.get_cahed_copula(right_bound, middle_lb)
                if copula_rl is None:
                    self.exec_count += 1
                    copula_rl = get_copula(
                        copula_type=self.get_copula_type(middle_lb, right_bound),
                        cdf1=get_cdf_value(right_bound),
                        cdf2=get_cdf_value(middle_lb),
                        theta=self.get_theta_value(middle_lb, right_bound),
                    )
                    self.store_copula(right_bound, middle_lb, value=copula_rl)

                copula_rr = self.get_cahed_copula(right_bound, middle_ub)
                if copula_rr is None:
                    self.exec_count += 1
                    copula_rr = get_copula(
                        copula_type=self.get_copula_type(middle_ub, right_bound),
                        cdf1=get_cdf_value(right_bound),
                        cdf2=get_cdf_value(middle_ub),
                        theta=self.get_theta_value(middle_ub, right_bound),
                    )
                    self.store_copula(right_bound, middle_ub, value=copula_rr)

                copula_left_diff = copula_lr - copula_ll
                copula_right_diff = copula_rr - copula_rl

                middle_p_value = get_cdf_value(middle_ub) - get_cdf_value(middle_lb)

                p_values[self.get_p_key(f"{middle_value}")] = middle_p_value
                p_values[self.get_p_key([left_bound, middle_value])] = copula_left_diff
                p_values[self.get_p_key([middle_value, right_bound])] = copula_right_diff

                if self.GRAD_ENABLED and middle_p_value <= 0:
                    grad_1 = gumbel_grad(get_cdf_value(left_bound), get_cdf_value(middle_ub), theta=self.get_theta_value(left_bound, middle_ub), copula=copula_ll) 
                    grad_2 = gumbel_grad(get_cdf_value(left_bound), get_cdf_value(middle_lb), theta=self.get_theta_value(left_bound, middle_ub), copula=copula_lr) 
                    grad_3 = gumbel_grad(get_cdf_value(right_bound), get_cdf_value(middle_ub), theta=self.get_theta_value(middle_lb, right_bound), copula=copula_rl)
                    grad_4 = gumbel_grad(get_cdf_value(right_bound), get_cdf_value(middle_lb), theta=self.get_theta_value(middle_lb, right_bound), copula=copula_rr)
                    copula_left_diff_div = grad_2 - grad_1
                    copula_right_diff_div = grad_4 - grad_3
                else:
                    middle_p_value = middle_p_value if middle_p_value > 0 else self.EPSILON
                    copula_left_diff_div = copula_left_diff / middle_p_value
                    copula_right_diff_div = copula_right_diff / middle_p_value

                # print(f"pass with {left_bound} - {right_bound}") if self.verbose else None
                # print("=" * 20, "Leaf end", "\n") if self.verbose else None
                self.exec_count += 1
                final_copula = get_copula(
                    copula_type=self.get_copula_type(left_bound, right_bound),
                    cdf1=copula_left_diff_div,
                    cdf2=copula_right_diff_div,
                    theta=self.get_theta_value(left_bound, right_bound),
                )

                print(f"Going outside... from {level} to {level-1} with value : {final_copula}") if self.verbose else None
                return final_copula

            else:

                """Left Bound"""
                l_middle_bound = middle_bound.copy()[:-1]
                right_column = middle_bound[-1]
                ll_right_bound = f"{right_column}_lb"
                lr_right_bound = f"{right_column}_ub"
                # print("Left Bound - 1") if self.verbose else None

                f1_p1 = self.get_cahed_copula(left_bound, l_middle_bound, ll_right_bound)
                if f1_p1 is None:
                    f1_p1 = compute(left_bound, l_middle_bound, ll_right_bound, level=level + 1)
                    self.store_copula(left_bound, l_middle_bound, ll_right_bound, value=f1_p1)
                # print("Left Bound - 2") if self.verbose else None

                f1_p2 = self.get_cahed_copula(left_bound, l_middle_bound, lr_right_bound)
                if f1_p2 is None:
                    f1_p2 = compute(left_bound, l_middle_bound, lr_right_bound, level=level + 1)
                    self.store_copula(left_bound, l_middle_bound, lr_right_bound, value=f1_p2)

                value, bound = lr_right_bound.split("_")
                if bound == "ub":
                    p_values[self.get_p_key(l_middle_bound, value)] = (
                        p_values[self.get_p_key(l_middle_bound, f"{value}_ub")]
                        - p_values[self.get_p_key(l_middle_bound, f"{value}_lb")]
                    )

                f1_p3_m = p_values[self.get_p_key(l_middle_bound)]
                f1_p3_d = p_values[self.get_p_key(middle_bound)]
                p_values[self.get_p_key(middle_bound, left_bound)] = (f1_p1 - f1_p2) * f1_p3_m

                f1_p3_d = f1_p3_d if f1_p3_d > 0 else self.EPSILON
                f1 = (f1_p2 - f1_p1) * f1_p3_m / f1_p3_d

                """Right Bound"""
                r_middle_bound = middle_bound.copy()[1:]
                left_column = middle_bound[0]
                rl_left_bound = f"{left_column}_lb"
                rr_left_bound = f"{left_column}_ub"
                # print("Right Bound - 1") if self.verbose else None
                f2_p1 = self.get_cahed_copula(rl_left_bound, r_middle_bound, right_bound)
                if f2_p1 is None:
                    f2_p1 = compute(rl_left_bound, r_middle_bound, right_bound, level=level + 1)
                    self.store_copula(rl_left_bound, r_middle_bound, right_bound, value=f2_p1)
                # print("Right Bound - 2") if self.verbose else None

                f2_p2 = self.get_cahed_copula(rr_left_bound, r_middle_bound, right_bound)
                if f2_p2 is None:
                    f2_p2 = compute(rr_left_bound, r_middle_bound, right_bound, level=level + 1)
                    self.store_copula(rr_left_bound, r_middle_bound, right_bound, value=f2_p2)

                f2_p3_m = p_values[self.get_p_key(r_middle_bound)]
                f2_p3_d = p_values[self.get_p_key(middle_bound)]
                p_values[self.get_p_key(middle_bound, right_bound)] = (f2_p2 - f2_p1) * f2_p3_m

                if self.GRAD_ENABLED and f2_p3_d <= 0:
                    grad_1 = gumbel_grad(get_cdf_value(rl_left_bound), get_cdf_value(right_bound), theta=self.get_theta_value(rl_left_bound, right_bound), copula=f2_p1) 
                    grad_2 = gumbel_grad(get_cdf_value(rr_left_bound), get_cdf_value(right_bound), theta=self.get_theta_value(rr_left_bound, right_bound), copula=f2_p2) 
                    f2 = (grad_2 - grad_1) * f2_p3_m
                else:
                    f2_p3_d = f2_p3_d if f2_p3_d > 0 else self.EPSILON
                    f2 = (f2_p2 - f2_p1) * f2_p3_m / f2_p3_d

                value, bound = right_bound.split("_")
                if bound == "ub":
                    p_values[self.get_p_key(middle_bound, value)] = (
                        p_values[self.get_p_key(middle_bound, f"{value}_ub")]
                        - p_values[self.get_p_key(middle_bound, f"{value}_lb")]
                    )

                result = get_copula(
                    copula_type=self.get_copula_type(left_bound, right_bound),
                    cdf1=f1,
                    cdf2=f2,
                    theta=self.get_theta_value(left_bound, right_bound),
                )
                print(f"Going outside... from {level} to {level-1} with value {result}") if self.verbose else None
                self.exec_count += 1
                return result

        result = 0
        for idx, pair in enumerate([["lb", "lb"], ["lb", "ub"], ["ub", "ub"], ["ub", "lb"]]):
            value = compute(f"{it_left_bound}_{pair[0]}", it_middle_bound.copy(), f"{it_right_bound}_{pair[1]}")
            result += ((-1) ** idx) * value
            logger.info(f"Pair : {pair} Value {value} | Result : {result}") if self.verbose else None

        final_result = result * p_values[self.get_p_key(it_middle_bound)]
        logger.info(f"Final Result : {final_result}") if self.verbose else None
        return final_result


def test_1():
    import timeit

    import numpy as np

    np.random.seed(42)
    div = DivineCopulaDynamicRecursive()
    div.verbose = True
    # lb_ub_list = list(np.random.randn(10))
    lb_ub_list = [0.01 + 0.01*i for i in range(10)]

    def my_function():
        return div.predict(lb_ub_list, column_list=[1, 2, 3, 4, 5])

    ret = my_function()
    print(f"Return : {ret}")

    # execution_time = timeit.timeit('my_function()', globals=globals(), number=10) / 10
    # print(f"Average execution time: {execution_time} seconds")


def test_2():
    theta_dict = {
        "1,2": 1.030689091488632,
        "1,3": 0.901542136374699,
        "1,4": 1.2426725132311551,
        "1,5": 1.0625242982273202,
        "1,6": 1.3082247814542334,
        "1,7": 1.0090071942625467,
        "1,8": 1.1142226947301925,
        "1,9": 1.0527585746125594,
        "1,10": 1.1156824952715052,
        "2,3": 1.0512920424277583,
        "2,4": 1.0030283472568617,
        "2,5": 1.0507749908113337,
        "2,6": 1.0131942799460616,
        "2,7": 0.8163015970643177,
        "2,8": 1.416827782058451,
        "2,9": 1.6375440951719216,
        "2,10": 0.9300643367833735,
        "3,4": 1.0131262617838044,
        "3,5": 1.264528910420137,
        "3,6": 0.8765641185456006,
        "3,7": 0.9175720914437737,
        "3,8": 0.7420120250266058,
        "3,9": 0.8697823185644987,
        "3,10": 0.8957406256511162,
        "4,5": 1.885602608013082,
        "4,6": 1.0328520831377053,
        "4,7": 0.9724113492701247,
        "4,8": 1.0190469295999145,
        "4,9": 1.0259278141406247,
        "4,10": 1.0522881598887681,
        "5,6": 0.9762760905051003,
        "5,7": 0.9188605878089624,
        "5,8": 0.9384843300064947,
        "5,9": 1.0270731620011195,
        "5,10": 0.9717064118784948,
        "6,7": 0.9933308629700952,
        "6,8": 1.133698453278433,
        "6,9": 1.0750267600302943,
        "6,10": 1.2842361530077,
        "7,8": 0.9283291294734797,
        "7,9": 0.602381022652425,
        "7,10": 1.0916074358516574,
        "8,9": 1.7176691118475125,
        "8,10": 1.0116065488838826,
        "9,10": 0.9474346984449047
    }

    theta_dict = {tuple(map(lambda x: int(x) - 1, k.split(","))): v for k, v in theta_dict.items()}
    div = DivineCopulaDynamicRecursive(theta_dict=theta_dict)
    
    lb_ub_list = [0.01 + 0.01*i for i in range(20)]

    def my_function():
        return div.predict(lb_ub_list, column_list=[i for i in range(1, 11)])

    ret = my_function()
    print(f"Return : {ret}")


if __name__ == "__main__":
    test_2()
    # test_1()