import sys
import logging
from pyomo.environ import *
import matplotlib.pyplot as plt
import math

logging.getLogger("pyomo.core").setLevel(logging.ERROR)

N = None
W = None
L = None
B = None
ALPHA = None
BETA = None
VMAX = None
n = None
cars = []
M = 10000
model = None
A = None


def processArgs():
    args = sys.argv[1:]
    global N, W, L, B, ALPHA, BETA, VMAX, n, cars, A
    N = int(args[0])
    W = int(args[1])
    L = int(args[2])
    B = int(args[3])
    ALPHA = int(args[4])
    BETA = int(args[5])
    VMAX = int(args[6])
    n = int(args[7])
    for i in range(n):
        cars.append(
            [
                float(args[8 + 4 * i]),
                float(args[9 + 4 * i]),
                int(args[10 + 4 * i]),
                int(args[11 + 4 * i]),
            ]
        )
    A = [ALPHA, 0, -BETA, 0] * N


def plots(model):
    plt.figure()
    for c in model.C:
        Ts = [0]
        Vs = [model.V0[c]]
        for j in range(4 * N):
            Ts.append(Ts[-1] + model.t[c, j].value)
            Vs.append(Vs[-1] + A[j] * model.t[c, j].value)
        plt.plot(Ts, Vs, label=f"Car {c}")
    plt.title(f"Velocity Profiles")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.legend()
    plt.savefig(
        "results/vprofiles_b.png",
    )
    plt.figure()
    for c in model.C:
        Ts = [0]
        Ss = [model.S0[c]]
        Vs = [model.V0[c]]
        for j in range(4 * N):
            last_T = Ts[-1]
            last_S = Ss[-1]
            for k in range(100):
                t = model.t[c, j].value * k / 100
                Ts.append(last_T + t)
                Ss.append(last_S - (Vs[-1] * t + 0.5 * A[j] * t * t))
            Vs.append(Vs[-1] + A[j] * model.t[c, j].value)
        plt.plot(Ts, Ss, label=f"Car {c}")
    plt.title(f"Distance Profiles")
    plt.xlabel("Time")
    plt.ylabel("Distance")
    plt.legend()
    plt.savefig(
        "results/sprofiles_b.png",
    )


if __name__ == "__main__":
    processArgs()

    model = ConcreteModel()

    # %% Params
    model.C = Set(ordered=True, initialize=range(n))

    def S0(model, i):
        return cars[i][0]

    model.S0 = Param(model.C, initialize=S0)

    def V0(model, i):
        return cars[i][1]

    model.V0 = Param(model.C, initialize=V0)

    def inLane(model, i):
        return cars[i][2]

    model.in_lane = Param(model.C, initialize=inLane)

    def turn(model, i):
        return cars[i][3]

    model.turn = Param(model.C, initialize=turn)

    # %% Variables
    model.t = Var(model.C, range(4 * N), within=NonNegativeReals)
    model.t_int = Var(model.C, within=NonNegativeReals)
    model.z = Var(model.C, model.C, range(2), within=Binary)

    # %% Objective
    def objective_rule(model):
        return sum((model.t_int[i]) for i in model.C) + sum(
            (sum((model.t[i, k]) for k in range(4 * N))) for i in model.C
        )

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # %% helper functions
    def getV(model, c, k):
        return model.V0[c] + sum((A[i] * model.t[c, i]) for i in range(k))

    def getS(model, c, k):
        return model.S0[c] - sum(
            (
                getV(model, c, i) * model.t[c, i]
                + 0.5 * A[i] * model.t[c, i] * model.t[c, i]
            )
            for i in range(k)
        )

    def getT(model, c, k):
        return sum(model.t[c, j] for j in range(k))

    # %% speed limit constraint
    def add_speed_limit_constraints(model):
        for c in model.C:
            for i in range(1, 4 * N + 1):
                model.speed_limit_constraints.add(getV(model, c, i) <= VMAX)
                model.speed_limit_constraints.add(getV(model, c, i) >= 0)

    model.speed_limit_constraints = ConstraintList()
    add_speed_limit_constraints(model)

    # %% ensuring car arrival at intersection
    def car_arrival_rule(model, c):
        return getS(model, c, 4 * N) == 0

    model.car_arrival_constraint = Constraint(model.C, rule=car_arrival_rule)

    # %% ensuring car exit from intersection
    def car_exit_rule(model, c):
        if model.turn[c] == 1:
            return getV(model, c, 4 * N) * model.t_int[c] == math.pi * (W + L) / 4
        elif model.turn[c] == 2:
            return getV(model, c, 4 * N) * model.t_int[c] == 2 * W + L
        else:
            return getV(model, c, 4 * N) * model.t_int[c] == math.pi * (3 * W + L) / 4

    model.car_exit_constraint = Constraint(model.C, rule=car_exit_rule)

    # %% LS collision constraints
    def add_LS_collision_constraints(model):
        for c1 in model.C:
            for c2 in model.C:
                if model.turn[c1] == 1 and model.turn[c2] == 2:
                    if model.in_lane[c1] == model.in_lane[c2]:
                        theta = math.acos((W + L - B) / (W + L + B))
                        phi = math.atan(L / (W + L + B))
                        T1_entry = getT(model, c1, 4 * N)
                        T2_entry = getT(model, c2, 4 * N)
                        T1_exit = T1_entry + model.t_int[c1] * ((W + L) / 2) * (
                            theta + phi
                        ) / (math.pi * (W + L) / 4)
                        T2_exit = T2_entry + model.t_int[c2] * (
                            (W + L + B) * math.tan(theta) / 2 + L / 2
                        ) / (2 * W + L)
                        model.LS_collision_constraints.add(
                            T1_exit - T2_entry <= M * (model.z[c1, c2, 0])
                        )
                        model.LS_collision_constraints.add(
                            T2_exit - T1_entry <= M * (model.z[c1, c2, 1])
                        )
                        model.LS_collision_constraints.add(
                            model.z[c1, c2, 0] + model.z[c1, c2, 1] == 1
                        )
                    elif (model.in_lane[c1] + 3) % 4 == model.in_lane[c2] % 4:
                        theta = math.acos((W + L - B) / (W + L + B))
                        phi = math.atan(L / (W + L + B))
                        T1_exit = getT(model, c1, 4 * N) + model.t_int[c1]
                        T2_exit = getT(model, c2, 4 * N) + model.t_int[c2]
                        T1_entry = T1_exit - model.t_int[c1] * ((W + L) / 2) * (
                            theta + phi
                        ) / (math.pi * (W + L) / 4)
                        T2_entry = T2_exit - model.t_int[c2] * (
                            (W + L + B) * math.tan(theta) / 2 + L / 2
                        ) / (2 * W + L)
                        model.LS_collision_constraints.add(
                            T1_exit - T2_entry <= M * (model.z[c1, c2, 0])
                        )
                        model.LS_collision_constraints.add(
                            T2_exit - T1_entry <= M * (model.z[c1, c2, 1])
                        )
                        model.LS_collision_constraints.add(
                            model.z[c1, c2, 0] + model.z[c1, c2, 1] == 1
                        )

    model.LS_collision_constraints = ConstraintList()
    add_LS_collision_constraints(model)

    # %% L-R collisions
    def add_LR_collision_constraints(model):
        for c1 in model.C:
            for c2 in model.C:
                if model.turn[c1] == 1 and model.turn[c2] == 3:
                    if model.in_lane[c1] == model.in_lane[c2]:
                        R1 = (W + L) / 2
                        R2 = (3 * W + L) / 2
                        phi1 = math.asin(L / (W + L + B))
                        phi2 = math.asin(L / (3 * W + L + B))
                        theta1 = 2 * math.atan(math.sqrt(B / (B + 2 * (R1 + R2))))
                        theta2 = math.asin(
                            math.sqrt(B * (B + 2 * R1 + 2 * R2))
                            * (B + 2 * R1)
                            / (B + 2 * R2)
                            / (B + R1 + R2)
                        )
                        T1_entry = getT(model, c1, 4 * N)
                        T2_entry = getT(model, c2, 4 * N)
                        T1_exit = getT(model, c1, 4 * N) + model.t_int[c1] * (
                            theta1 + phi1
                        ) / (math.pi / 2)
                        T2_exit = getT(model, c2, 4 * N) + model.t_int[c2] * (
                            theta2 + phi2
                        ) / (math.pi / 2)
                        model.LR_collision_constraints.add(
                            T1_exit - T2_entry <= M * (model.z[c1, c2, 0])
                        )
                        model.LR_collision_constraints.add(
                            T2_exit - T1_entry <= M * (model.z[c1, c2, 1])
                        )
                        model.LR_collision_constraints.add(
                            model.z[c1, c2, 0] + model.z[c1, c2, 1] == 1
                        )
                    elif (model.in_lane[c1] + 2) % 4 == model.in_lane[c2] % 4:
                        R1 = (W + L) / 2
                        R2 = (3 * W + L) / 2
                        phi1 = math.asin(L / (W + L + B))
                        phi2 = math.asin(L / (3 * W + L + B))
                        theta1 = 2 * math.atan(math.sqrt(B / (B + 2 * (R1 + R2))))
                        theta2 = math.asin(
                            math.sqrt(B * (B + 2 * R1 + 2 * R2))
                            * (B + 2 * R1)
                            / (B + 2 * R2)
                            / (B + R1 + R2)
                        )
                        T1_entry = getT(model, c1, 4 * N) + model.t_int[c1] * (
                            math.pi / 2 - theta1 - phi1
                        ) / (math.pi / 2)
                        T2_entry = getT(model, c2, 4 * N) + model.t_int[c2] * (
                            math.pi / 2 - theta2 - phi2
                        ) / (math.pi / 2)
                        T1_exit = getT(model, c1, 4 * N) + model.t_int[c1]
                        T2_exit = getT(model, c2, 4 * N) + model.t_int[c2]
                        model.LR_collision_constraints.add(
                            T1_exit - T2_entry <= M * (model.z[c1, c2, 0])
                        )
                        model.LR_collision_constraints.add(
                            T2_exit - T1_entry <= M * (model.z[c1, c2, 1])
                        )
                        model.LR_collision_constraints.add(
                            model.z[c1, c2, 0] + model.z[c1, c2, 1] == 1
                        )

    model.LR_collision_constraints = ConstraintList()
    add_LR_collision_constraints(model)

    # %% S-S collisions
    def add_SS_collision_constraints(model):
        for c1 in model.C:
            for c2 in model.C:
                if model.turn[c1] == 2 and model.turn[c2] == 2:
                    if (model.in_lane[c1] + 1) % 4 == model.in_lane[c2] % 4:
                        T1_entry = getT(model, c1, 4 * N) + model.t_int[c1] * (
                            (3 * W - B) / 2
                        ) / (2 * W + L)
                        T2_entry = getT(model, c2, 4 * N) + model.t_int[c2] * (
                            (W - B) / 2
                        ) / (2 * W + L)
                        T1_exit = T1_entry + model.t_int[c1] * (L + B) / (2 * W + L)
                        T2_exit = T2_entry + model.t_int[c2] * (L + B) / (2 * W + L)
                        model.SS_collision_constraints.add(
                            T1_exit - T2_entry <= M * (model.z[c1, c2, 0])
                        )
                        model.SS_collision_constraints.add(
                            T2_exit - T1_entry <= M * (model.z[c1, c2, 1])
                        )
                        model.SS_collision_constraints.add(
                            model.z[c1, c2, 0] + model.z[c1, c2, 1] == 1
                        )
                    # elif (model.in_lane[c1] == model.in_lane[c2]):

    model.SS_collision_constraints = ConstraintList()
    add_SS_collision_constraints(model)

    # %% S-R collision constraints
    def add_SR_collision_constraints(model):
        for c1 in model.C:
            for c2 in model.C:
                if model.turn[c1] == 2 and model.turn[c2] == 3:
                    if model.in_lane[c1] == model.in_lane[c2]:
                        T1_entry = getT(model, c1, 4 * N)
                        T2_entry = getT(model, c2, 4 * N)
                        theta = math.acos((3 * W + L - B) / (3 * W + L + B))
                        phi = math.asin(L / (3 * W + L + B))
                        T1_exit = T1_entry + model.t_int[c1] * (
                            (3 * W + L + B) * math.tan(theta) / 2 + L / 2
                        ) / (2 * W + L)
                        T2_exit = T2_entry + model.t_int[c2] * ((3 * W + L) / 2) * (
                            theta + phi
                        ) / (math.pi * (3 * W + L) / 4)
                        model.SR_collision_constraints.add(
                            T1_exit - T2_entry <= M * (model.z[c1, c2, 0])
                        )
                        model.SR_collision_constraints.add(
                            T2_exit - T1_entry <= M * (model.z[c1, c2, 1])
                        )
                        model.SR_collision_constraints.add(
                            model.z[c1, c2, 0] + model.z[c1, c2, 1] == 1
                        )
                    elif (model.in_lane[c1] + 1) % 4 == model.in_lane[c2] % 4:
                        theta1 = math.asin((W - B + L) / (3 * W + L + B))
                        theta2 = math.asin((W + B + L) / (3 * W + L - B))
                        phi = math.asin(L / (3 * W + L + B))
                        x1 = ((3 * W + L - B) / 2) * math.cos(theta2) - L / 2
                        x2 = ((3 * W + L + B) / 2) * math.cos(theta1) + L / 2
                        T1_entry = getT(model, c1, 4 * N) + model.t_int[c1] * x1 / (
                            2 * W + L
                        )
                        T2_entry = getT(model, c2, 4 * N) + model.t_int[c2] * (
                            theta1 - phi
                        ) / (math.pi / 2)
                        T1_exit = getT(model, c1, 4 * N) + model.t_int[c1] * x2 / (
                            2 * W + L
                        )
                        T2_exit = getT(model, c2, 4 * N) + model.t_int[c2] * (
                            theta2 + phi
                        ) / (math.pi / 2)
                        model.SR_collision_constraints.add(
                            T1_exit - T2_entry <= M * (model.z[c1, c2, 0])
                        )
                        model.SR_collision_constraints.add(
                            T2_exit - T1_entry <= M * (model.z[c1, c2, 1])
                        )
                        model.SR_collision_constraints.add(
                            model.z[c1, c2, 0] + model.z[c1, c2, 1] == 1
                        )
                    elif (model.in_lane[c1] + 2) % 4 == model.in_lane[c2] % 4:
                        theta2 = math.acos((W - B + L) / (3 * W + L + B))
                        theta1 = math.acos((W + B + L) / (3 * W + L - B))
                        phi = math.asin(L / (3 * W + L + B))
                        x2 = (2 * W + L / 2) - (3 * W + L + B) * math.sin(theta2) / 2
                        x1 = (2 * W + 3 * L / 2) - (3 * W + L - B) * math.sin(
                            theta1
                        ) / 2
                        T1_entry = getT(model, c1, 4 * N) + model.t_int[c1] * (x2) / (
                            2 * W + L
                        )
                        T1_exit = getT(model, c1, 4 * N) + model.t_int[c1] * (x1) / (
                            2 * W + L
                        )
                        T2_entry = getT(model, c2, 4 * N) + model.t_int[c2] * (
                            theta1 - phi
                        ) / (math.pi / 2)
                        T2_exit = getT(model, c2, 4 * N) + model.t_int[c2] * (
                            theta2 + phi
                        ) / (math.pi / 2)
                        model.SR_collision_constraints.add(
                            T1_exit - T2_entry <= M * (model.z[c1, c2, 0])
                        )
                        model.SR_collision_constraints.add(
                            T2_exit - T1_entry <= M * (model.z[c1, c2, 1])
                        )
                        model.SR_collision_constraints.add(
                            model.z[c1, c2, 0] + model.z[c1, c2, 1] == 1
                        )
                    elif (model.in_lane[c1] + 3) % 4 == model.in_lane[c2] % 4:
                        phi = math.asin(L / (3 * W + L + B))
                        theta = math.acos((3 * W + L - B) / (3 * W + L + B))
                        T1_exit = getT(model, c1, 4 * N) + model.t_int[c1]
                        T2_exit = getT(model, c2, 4 * N) + model.t_int[c2]
                        x = (2 * W + L) - (
                            (3 * W + L + B) * math.sin(theta) / 2 + L / 2
                        )
                        T1_entry = getT(model, c1, 4 * N) - model.t_int[c1] * x / (
                            2 * W + L
                        )
                        T2_entry = T2_exit - model.t_int[c2] * (theta + phi) / (
                            math.pi / 2
                        )
                        model.SR_collision_constraints.add(
                            T1_exit - T2_entry <= M * (model.z[c1, c2, 0])
                        )
                        model.SR_collision_constraints.add(
                            T2_exit - T1_entry <= M * (model.z[c1, c2, 1])
                        )
                        model.SR_collision_constraints.add(
                            model.z[c1, c2, 0] + model.z[c1, c2, 1] == 1
                        )

    model.SR_collision_constraints = ConstraintList()
    add_SR_collision_constraints(model)

    # %% R-R collision constraints
    def add_RR_collision_constraints(model):
        for c1 in model.C:
            for c2 in model.C:
                if model.turn[c1] == 3 and model.turn[c2] == 3:
                    if (model.in_lane[c1] + 1) % 4 == model.in_lane[c2] % 4:
                        R = (3 * W + L) / 2
                        phi = math.asin(L / (3 * W + L + B))
                        theta1 = 2 * math.atan(
                            math.sqrt(
                                ((B - L - 2 * W) * (L - 2 * R + 2 * W))
                                / ((B + L + 2 * W) * (L + 2 * R + 2 * W))
                            )
                        )
                        theta2 = 2 * math.atan(
                            math.cos(theta1 / 2) / math.sin(theta1 / 2)
                        )
                        T1_entry = getT(model, c1, 4 * N) + model.t_int[c1] * (
                            theta1 - phi
                        ) / (math.pi / 2)
                        T2_entry = getT(model, c2, 4 * N) + model.t_int[c2] * (
                            math.pi / 2 - theta2 - phi
                        ) / (math.pi / 2)
                        T1_exit = getT(model, c1, 4 * N) + model.t_int[c1] * (
                            theta2 + phi
                        ) / (math.pi / 2)
                        T2_exit = getT(model, c2, 4 * N) + model.t_int[c2] * (
                            math.pi / 2 - theta1 + phi
                        ) / (math.pi / 2)
                        model.RR_collision_constraints.add(
                            T1_exit - T2_entry <= M * (model.z[c1, c2, 0])
                        )
                        model.RR_collision_constraints.add(
                            T2_exit - T1_entry <= M * (model.z[c1, c2, 1])
                        )
                        model.RR_collision_constraints.add(
                            model.z[c1, c2, 0] + model.z[c1, c2, 1] == 1
                        )
                    elif (model.in_lane[c1] + 2) % 4 == model.in_lane[c2] % 4:
                        phi = math.asin(L / (3 * W + L + B))
                        theta = 0.5 * math.asin(
                            ((4 * W + 2 * L) / (3 * W + L + B))
                            * ((4 * W + 2 * L) / (3 * W + L + B))
                            - 1
                        )
                        T1_entry = getT(model, c1, 4 * N) + model.t_int[c1] * (
                            theta - phi
                        ) / (math.pi / 2)
                        T2_entry = getT(model, c2, 4 * N) + model.t_int[c2] * (
                            theta - phi
                        ) / (math.pi / 2)
                        T1_exit = getT(model, c1, 4 * N) + model.t_int[c1] * (
                            math.pi / 2 - theta + phi
                        ) / (math.pi / 2)
                        T2_exit = getT(model, c2, 4 * N) + model.t_int[c2] * (
                            math.pi / 2 - theta + phi
                        ) / (math.pi / 2)
                        model.RR_collision_constraints.add(
                            T1_exit - T2_entry <= M * (model.z[c1, c2, 0])
                        )
                        model.RR_collision_constraints.add(
                            T2_exit - T1_entry <= M * (model.z[c1, c2, 1])
                        )
                        model.RR_collision_constraints.add(
                            model.z[c1, c2, 0] + model.z[c1, c2, 1] == 1
                        )

    model.RR_collision_constraints = ConstraintList()
    add_RR_collision_constraints(model)

    # %% solving using gurobi

    timeout_seconds = 60
    opt = SolverFactory("gurobi")
    opt.options["TimeLimit"] = timeout_seconds
    try:
        results = opt.solve(model)
        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            print("SOLVED", end=" ")
            for c in model.C:
                for i in range(0, 4 * N):
                    print(model.t[c, i].value, end=" ")
                print(model.t_int[c].value, end="\n")
            plots(model)

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            print("INFEASIBLE")
        elif results.solver.status == SolverStatus.aborted:
            print("ABORTED")
    except:
        print("TIMEOUT")
