
## Ethics Aware Resource Distribution — Card Dashboard UI

### Overview
Rebuild the simulator with a light, warm aesthetic using the Card Dashboard style: stone/amber color palette, rounded cards, agent avatar badges, and clean typography.

### Layout (3-column on desktop, stacked on mobile)
- **Left sidebar (col-span-3):** Controls card (rounds input, resources input, Run Simulation button) + Info/Definitions card with colored dot indicators
- **Main area (col-span-9):** Scrollable round cards displayed newest-first, each with a data table showing Agent, Req, Alloc, Waiting, Fulfilled, Bid columns
- **Header:** App title with amber icon + "Total Resource: 16" subtitle

### Visual Details
- Background: stone-50, cards: white with stone-200 borders and rounded-2xl
- Agent rows with numbered amber avatar circles
- Alloc column: green for allocated, red for zero
- Bid values in blue monospace
- "Latest" badge on most recent round
- Hover state on table rows
- Run Simulation button: stone-900 bg, white text

### Simulation Logic
- Replicate the ethical fairness algorithm: 5 agents compete for limited resources over N rounds
- Priority/bid evolves based on past allocation history (waiting time increases priority)
- Each round: agents request random amounts, allocation based on bid strength, unfulfilled agents gain priority
- State managed with React useState; results displayed as round cards

### Components
1. **SimulationPage** (index.tsx) — main page with state management and simulation logic
2. **ControlsPanel** — inputs for rounds/resources + run button
3. **RoundCard** — displays one round's allocation table
4. **DefinitionsPanel** — info sidebar with definitions and credits

### Responsive
- On mobile: stack all panels vertically (controls → rounds → definitions)
