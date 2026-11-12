
import csv
import json
import math
import random
import statistics
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# ---------------- SETTINGS ----------------

CAPACITY = 100
SERVICE = 10
HORIZON = 700
VEHICLE_COST = 30
LATE_PENALTY = 10

PTS, DEMAND, DIST, READY, DUE = [], [], [], [], []
CUSTOMERS = []
RESULTS = {}
ROUTES = {}
HISTORY = {}


# ---------------- DATASET ----------------

def load_dataset(filename):
    global PTS, DEMAND, DIST, READY, DUE, CUSTOMERS

    with open(filename, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"id", "x", "y", "demand", "ready", "due"}

        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "CSV columns must be: id,x,y,demand,ready,due"
            )

        rows = list(reader)

    if len(rows) < 2:
        raise ValueError("Dataset must contain a depot and customers.")

    rows.sort(key=lambda r: int(r["id"]))

    if [int(r["id"]) for r in rows] != list(range(len(rows))):
        raise ValueError("IDs must start at 0 and continue without gaps.")

    points, demands, ready, due = [], [], [], []

    for row in rows:
        points.append((float(row["x"]), float(row["y"])))
        demands.append(int(row["demand"]))
        ready.append(float(row["ready"]))
        due.append(float(row["due"]))

    if demands[0] != 0:
        raise ValueError("Depot (ID 0) must have demand 0.")

    if any(d <= 0 for d in demands[1:]):
        raise ValueError("Customer demands must be positive.")

    if any(d > CAPACITY for d in demands[1:]):
        raise ValueError("A customer demand exceeds vehicle capacity.")

    if any(ready[i] > due[i] for i in range(len(rows))):
        raise ValueError("Ready time cannot exceed due time.")

    if any(r < 0 or d > HORIZON for r, d in zip(ready, due)):
        raise ValueError("Time windows must be within the planning horizon.")

    PTS = points
    DEMAND = demands
    READY = ready
    DUE = due
    CUSTOMERS = list(range(1, len(points)))

    DIST = [
        [
            math.hypot(a[0] - b[0], a[1] - b[1])
            for b in PTS
        ]
        for a in PTS
    ]

    return len(CUSTOMERS)


# ---------------- ROUTE DECODING ----------------

def decode(perm):
    routes = []
    route = []
    load = 0
    time = 0.0
    last = 0

    for customer in perm:
        arrival = time + DIST[last][customer]
        start = max(arrival, READY[customer])

        feasible = (
            load + DEMAND[customer] <= CAPACITY
            and start <= DUE[customer]
            and start + SERVICE + DIST[customer][0] <= HORIZON
        )

        if route and not feasible:
            routes.append(route)
            route = []
            load = 0
            time = 0.0
            last = 0

            arrival = DIST[0][customer]
            start = max(arrival, READY[customer])

        # Keep the customer in a route even if individually infeasible.
        # The evaluator will report the resulting constraint violations.
        route.append(customer)
        load += DEMAND[customer]
        time = start + SERVICE
        last = customer

    if route:
        routes.append(route)

    return routes


def evaluate(routes):
    distance = 0.0
    lateness = 0.0
    violations = 0

    for route in routes:
        time = 0.0
        load = 0
        last = 0

        for customer in route:
            arrival = time + DIST[last][customer]
            start = max(arrival, READY[customer])

            if start > DUE[customer]:
                lateness += start - DUE[customer]
                violations += 1

            load += DEMAND[customer]

            if load > CAPACITY:
                violations += 1

            distance += DIST[last][customer]
            time = start + SERVICE
            last = customer

        distance += DIST[last][0]

        if time + DIST[last][0] > HORIZON:
            violations += 1

    return distance, lateness, violations


def fitness(perm):
    routes = decode(perm)
    distance, lateness, violations = evaluate(routes)

    # Large penalty discourages constraint violations.
    return (
        distance
        + VEHICLE_COST * len(routes)
        + LATE_PENALTY * lateness
        + 100000 * violations
    )


def summary(perm):
    routes = decode(perm)
    distance, lateness, violations = evaluate(routes)

    return {
        "customers": len(CUSTOMERS),
        "vehicles": len(routes),
        "distance": round(distance, 2),
        "lateness": round(lateness, 2),
        "violations": violations,
        "fitness": round(fitness(perm), 2)
    }


# ---------------- NEAREST NEIGHBOR ----------------

def nearest_neighbor(randomize=0, rnd=None):
    rnd = rnd or random.Random(42)
    unvisited = set(CUSTOMERS)
    permutation = []

    while unvisited:
        route, load, time, last = [], 0, 0.0, 0

        while True:
            candidates = []

            for customer in unvisited:
                arrival = time + DIST[last][customer]
                start = max(arrival, READY[customer])

                if (
                    load + DEMAND[customer] <= CAPACITY
                    and start <= DUE[customer]
                    and start + SERVICE + DIST[customer][0] <= HORIZON
                ):
                    score = (
                        DIST[last][customer]
                        + 0.5 * max(0, READY[customer] - arrival)
                    )
                    candidates.append((score, customer))

            if not candidates:
                break

            candidates.sort()
            count = max(1, randomize)
            chosen = (
                rnd.choice(candidates[:count])[1]
                if randomize else candidates[0][1]
            )

            arrival = time + DIST[last][chosen]
            time = max(arrival, READY[chosen]) + SERVICE
            load += DEMAND[chosen]
            last = chosen

            permutation.append(chosen)
            unvisited.remove(chosen)
            route.append(chosen)

        if not route:
            # Avoid an infinite loop for an infeasible dataset.
            chosen = min(unvisited)
            permutation.append(chosen)
            unvisited.remove(chosen)

    return permutation


# ---------------- GENETIC ALGORITHM ----------------

def tournament(population, scores, rnd, k=3):
    indexes = rnd.sample(range(len(population)), min(k, len(population)))
    winner = min(indexes, key=lambda i: scores[i])
    return population[winner][:]


def crossover(p1, p2, rnd):
    n = len(p1)

    if n < 2:
        return p1[:]

    a, b = sorted(rnd.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]

    used = set(child[a:b + 1])
    fill = [gene for gene in p2 if gene not in used]
    j = 0

    for i in range(n):
        if child[i] is None:
            child[i] = fill[j]
            j += 1

    return child


def mutate(permutation, rnd):
    child = permutation[:]

    if len(child) < 2:
        return child

    a, b = sorted(rnd.sample(range(len(child)), 2))

    if rnd.random() < 0.5:
        child[a], child[b] = child[b], child[a]
    else:
        child[a:b + 1] = reversed(child[a:b + 1])

    return child


def run_ga(seed, hybrid, pop_size=60, generations=150):
    rnd = random.Random(seed)
    population = []

    if hybrid:
        population.append(nearest_neighbor(0, rnd))

        for _ in range(max(0, int(pop_size * 0.2) - 1)):
            population.append(nearest_neighbor(3, rnd))

    while len(population) < pop_size:
        individual = CUSTOMERS[:]
        rnd.shuffle(individual)
        population.append(individual)

    scores = [fitness(p) for p in population]
    best_index = min(range(len(scores)), key=lambda i: scores[i])
    best_score = scores[best_index]
    best_permutation = population[best_index][:]
    history = [best_score]
    stalled = 0

    for _ in range(generations):
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        new_population = [population[i][:] for i in order[:2]]

        while len(new_population) < pop_size:
            p1 = tournament(population, scores, rnd)
            p2 = tournament(population, scores, rnd)

            child = crossover(p1, p2, rnd) if rnd.random() < 0.9 else p1

            if rnd.random() < 0.3:
                child = mutate(child, rnd)

            new_population.append(child)

        population = new_population
        scores = [fitness(p) for p in population]

        index = min(range(len(scores)), key=lambda i: scores[i])

        if scores[index] < best_score:
            best_score = scores[index]
            best_permutation = population[index][:]
            stalled = 0
        else:
            stalled += 1

        history.append(best_score)

        if stalled >= 40:
            break

    return best_score, best_permutation, history


# ---------------- GRAPH FUNCTIONS ----------------

def show_routes(permutation, title):
    routes = decode(permutation)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, route in enumerate(routes):
        xs = [PTS[0][0]] + [PTS[c][0] for c in route] + [PTS[0][0]]
        ys = [PTS[0][1]] + [PTS[c][1] for c in route] + [PTS[0][1]]
        ax.plot(xs, ys, "-o", markersize=4, label=f"Vehicle {i + 1}")

    ax.scatter(
        PTS[0][0], PTS[0][1],
        marker="s", s=150, color="red", label="Depot"
    )

    ax.set_title(title)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def show_convergence():
    if not HISTORY:
        messagebox.showwarning("Warning", "Run the algorithms first.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, values in HISTORY.items():
        ax.plot(values, label=name)

    ax.set_title("Algorithm Convergence")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


# ---------------- APPLICATION ----------------

class VRPApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Routing Problem - Optimization System")
        self.root.geometry("950x650")
        self.root.minsize(800, 550)

        self.dataset_loaded = False
        self.status = tk.StringVar(value="Please load your CSV dataset.")
        self.best_permutations = {}

        title = ttk.Label(
            root,
            text="VEHICLE ROUTING OPTIMIZATION SYSTEM",
            font=("Arial", 19, "bold")
        )
        title.pack(pady=15)

        toolbar = ttk.Frame(root)
        toolbar.pack(fill="x", padx=20, pady=5)

        ttk.Button(
            toolbar, text="Load Dataset",
            command=self.load_file
        ).pack(side="left", padx=5)

        self.run_button = ttk.Button(
            toolbar, text="Run Algorithms",
            command=self.run_algorithms,
            state="disabled"
        )
        self.run_button.pack(side="left", padx=5)

        self.route_button = ttk.Button(
            toolbar, text="View Best Route",
            command=self.view_route,
            state="disabled"
        )
        self.route_button.pack(side="left", padx=5)

        self.graph_button = ttk.Button(
            toolbar, text="Convergence Graph",
            command=show_convergence,
            state="disabled"
        )
        self.graph_button.pack(side="left", padx=5)

        self.export_button = ttk.Button(
            toolbar, text="Export JSON",
            command=self.export_results,
            state="disabled"
        )
        self.export_button.pack(side="left", padx=5)

        ttk.Label(
            root, textvariable=self.status,
            foreground="blue"
        ).pack(pady=8)

        columns = (
            "algorithm", "vehicles", "distance",
            "lateness", "violations", "fitness"
        )

        self.table = ttk.Treeview(
            root, columns=columns, show="headings", height=12
        )

        headings = {
            "algorithm": "Algorithm",
            "vehicles": "Vehicles",
            "distance": "Distance",
            "lateness": "Lateness",
            "violations": "Violations",
            "fitness": "Fitness"
        }

        for col in columns:
            self.table.heading(col, text=headings[col])
            self.table.column(col, width=135, anchor="center")

        self.table.pack(fill="both", expand=True, padx=20, pady=10)

        ttk.Label(
            root,
            text=(
                "Dataset columns: id, x, y, demand, ready, due"
                "   |   Depot ID = 0"
            )
        ).pack(pady=5)

    def load_file(self):
        filename = filedialog.askopenfilename(
            title="Select customer dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            count = load_dataset(filename)
            self.dataset_loaded = True
            self.status.set(
                f"Loaded {count} customers successfully. "
                f"Depot: {PTS[0]}"
            )
            self.run_button.config(state="normal")
            messagebox.showinfo(
                "Dataset Loaded",
                f"{count} customers loaded successfully."
            )
        except Exception as error:
            self.dataset_loaded = False
            self.run_button.config(state="disabled")
            messagebox.showerror("Dataset Error", str(error))

    def run_algorithms(self):
        if not self.dataset_loaded:
            messagebox.showwarning("Warning", "Load a dataset first.")
            return

        self.run_button.config(state="disabled")
        self.status.set("Running algorithms. Please wait...")
        threading.Thread(
            target=self.calculate, daemon=True
        ).start()

    def calculate(self):
        try:
            local_results = {}
            local_permutations = {}
            local_history = {}

            nn = nearest_neighbor()
            local_results["Nearest Neighbor"] = summary(nn)
            local_permutations["Nearest Neighbor"] = nn

            for name, hybrid in [
                ("GA", False),
                ("Hybrid GA", True)
            ]:
                runs = [
                    run_ga(seed, hybrid)
                    for seed in range(3)
                ]

                best = min(runs, key=lambda result: result[0])
                scores = [result[0] for result in runs]

                details = summary(best[1])
                details["mean_fitness"] = round(
                    statistics.mean(scores), 2
                )
                details["std_fitness"] = round(
                    statistics.pstdev(scores), 2
                )

                local_results[name] = details
                local_permutations[name] = best[1]
                local_history[name] = best[2]

            self.root.after(
                0, lambda: self.finish(
                    local_results,
                    local_permutations,
                    local_history
                )
            )

        except Exception as error:
            self.root.after(
                0, lambda: self.failed(str(error))
            )

    def finish(self, results, permutations, history):
        global RESULTS, ROUTES, HISTORY

        RESULTS = results
        ROUTES = {
            name: decode(perm)
            for name, perm in permutations.items()
        }
        HISTORY = history
        self.best_permutations = permutations

        for item in self.table.get_children():
            self.table.delete(item)

        for name, data in results.items():
            self.table.insert(
                "", "end",
                values=(
                    name,
                    data["vehicles"],
                    data["distance"],
                    data["lateness"],
                    data["violations"],
                    data["fitness"]
                )
            )

        self.status.set("All algorithms completed.")
        self.run_button.config(state="normal")
        self.route_button.config(state="normal")
        self.graph_button.config(state="normal")
        self.export_button.config(state="normal")

    def failed(self, error):
        self.status.set("Algorithm execution failed.")
        self.run_button.config(state="normal")
        messagebox.showerror("Execution Error", error)

    def view_route(self):
        if not self.best_permutations:
            return

        name = min(
            self.best_permutations,
            key=lambda n: RESULTS[n]["fitness"]
        )
        show_routes(
            self.best_permutations[name],
            f"Best Route - {name}"
        )

    def export_results(self):
        if not RESULTS:
            return

        filename = filedialog.asksaveasfilename(
            title="Save results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )

        if not filename:
            return

        try:
            with open(filename, "w", encoding="utf-8") as file:
                json.dump(RESULTS, file, indent=4)

            messagebox.showinfo(
                "Export Successful",
                "Results saved successfully."
            )
        except Exception as error:
            messagebox.showerror("Export Error", str(error))


if __name__ == "__main__":
    root = tk.Tk()
    app = VRPApplication(root)
    root.mainloop()

