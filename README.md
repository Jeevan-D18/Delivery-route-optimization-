# Delivery Route Optimization – Hybrid Genetic Algorithm (3D)

React + Vite + Three.js (React Three Fiber, Drei) + Tailwind + Recharts + Lucide.

```
npm install
npm run dev      # http://localhost:5173
npm run build
```

- CSV columns: `id,x,y,demand,ready,due,service,depot` (depot = 1 for exactly one row). "Sample" button downloads a template.
- Vehicle capacity is a user setting. The demo dataset uses 80 (labelled demo parameter); uploaded data requires you to set it.
- GA: permutation chromosome, split into routes by capacity; fitness = distance + 30 × vehicles + 10 × lateness; tournament selection, order crossover, swap mutation, elitism (2), 300 generations.
- Hybrid GA = 1 nearest-neighbour seed + random individuals; Plain GA = random only. Both run live side by side.
- Experiments tab values are the recorded report numbers and are not recomputed from uploaded data.
- X,Y values are plane units; the simulation is not GPS or road-network tracking.
