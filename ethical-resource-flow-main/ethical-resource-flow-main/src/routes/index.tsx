import { createFileRoute } from "@tanstack/react-router";
import { useState, useEffect, useCallback } from "react";
import { ControlsPanel } from "@/components/ControlsPanel";
import { DefinitionsPanel } from "@/components/DefinitionsPanel";
import { RoundCard, type AgentRoundData } from "@/components/RoundCard";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "Ethics Aware Resource Distribution Simulator" },
      {
        name: "description",
        content: "Simulate ethical fairness-based resource allocation among competing agents",
      },
    ],
  }),
});

interface RoundResult {
  round: number;
  agents: AgentRoundData[];
}

interface BackendAgentData {
  name: string;
  waiting: number;
  fullfilled: number;
  bid: number;
}

interface BackendPayload {
  round: number;
  request: number[];
  allocation: number[];
  agents: BackendAgentData[];
}

function mapPayloadToRoundResult(payload: BackendPayload): RoundResult {
  return {
    round: payload.round,
    agents: payload.agents.map((agent, index) => ({
      id: index + 1,
      request: payload.request[index] ?? 0,
      allocation: payload.allocation[index] ?? 0,
      waiting: agent.waiting,
      fulfilled: agent.fullfilled,
      bid: agent.bid,
    })),
  };
}

function Index() {
  const [rounds, setRounds] = useState(5);
  const [interval, setInterval] = useState(1);
  const [results, setResults] = useState<RoundResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [socketStatus, setSocketStatus] = useState("connecting");

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8765");

    ws.onopen = () => setSocketStatus("connected");
    ws.onerror = () => setSocketStatus("error");
    ws.onclose = () => setSocketStatus("closed");
    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data) as BackendPayload;
      setResults((prev) => [...prev, mapPayloadToRoundResult(payload)]);
    };

    return () => {
      ws.close();
    };
  }, []);

  const handleRun = useCallback(() => {
    setIsRunning(true);
    setResults([]);

    fetch("http://localhost:8001/run", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ rounds, interval }),
    })
      .then((res) => res.text())
      .then(() => {
        setIsRunning(false);
      })
      .catch(() => {
        setIsRunning(false);
      });
  }, [rounds, interval]);

  return (
    <div className="min-h-screen bg-stone-50">
      {/* Header */}
      <header className="border-b border-stone-200 bg-white">
        <div className="mx-auto max-w-7xl px-4 py-5 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-100">
              <svg
                className="h-5 w-5 text-amber-600"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={2}
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5m.75-9l3-3 2.148 2.148A12.061 12.061 0 0116.5 7.605"
                />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold text-stone-900 tracking-tight">
                Ethics Aware Resource Distribution
              </h1>
              <p className="text-sm text-stone-500">
                Total Resource: <span className="font-semibold text-stone-700">16</span>
                <span className="ml-3 text-xs uppercase tracking-wide text-stone-500">
                  {socketStatus}
                </span>
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
          {/* Left Sidebar */}
          <aside className="lg:col-span-3 space-y-4">
            <ControlsPanel
              rounds={rounds}
              interval={interval}
              onRoundsChange={setRounds}
              onIntervalChange={setInterval}
              onRun={handleRun}
              isRunning={isRunning}
            />
            <div className="hidden lg:block">
              <DefinitionsPanel />
            </div>
          </aside>

          {/* Main Results Area */}
          <section className="lg:col-span-9 space-y-4">
            {results.length === 0 ? (
              <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-stone-200 bg-white py-20 text-center">
                <div className="flex h-14 w-14 items-center justify-center rounded-full bg-stone-100 mb-4">
                  <svg
                    className="h-6 w-6 text-stone-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    strokeWidth={1.5}
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z"
                    />
                  </svg>
                </div>
                <p className="text-stone-500 text-sm">
                  Configure rounds and interval, then run the simulation.
                </p>
              </div>
            ) : (
              [...results]
                .reverse()
                .map((r, i) => (
                  <RoundCard
                    key={r.round}
                    roundNumber={r.round}
                    agents={r.agents}
                    isLatest={i === 0}
                  />
                ))
            )}
          </section>

          {/* Definitions on mobile */}
          <div className="lg:hidden">
            <DefinitionsPanel />
          </div>
        </div>
      </main>
    </div>
  );
}
