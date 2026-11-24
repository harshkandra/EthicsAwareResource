import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";

interface ControlsPanelProps {
  rounds: number;
  interval: number;
  onRoundsChange: (v: number) => void;
  onIntervalChange: (v: number) => void;
  onRun: () => void;
  isRunning: boolean;
}

export function ControlsPanel({
  rounds,
  interval,
  onRoundsChange,
  onIntervalChange,
  onRun,
  isRunning,
}: ControlsPanelProps) {
  return (
    <Card className="border-stone-200 bg-white rounded-2xl shadow-sm">
      <CardHeader className="pb-4">
        <CardTitle className="text-stone-800 text-lg">Controls</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="rounds" className="text-stone-600 text-sm">
            Number of Rounds
          </Label>
          <Input
            id="rounds"
            type="number"
            min={1}
            max={50}
            value={rounds}
            onChange={(e) => onRoundsChange(Number(e.target.value))}
            className="border-stone-200 bg-stone-50 focus-visible:ring-amber-400"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="interval" className="text-stone-600 text-sm">
            Interval (seconds)
          </Label>
          <Input
            id="interval"
            type="number"
            min={1}
            max={20}
            value={interval}
            onChange={(e) => onIntervalChange(Number(e.target.value))}
            className="border-stone-200 bg-stone-50 focus-visible:ring-amber-400"
          />
        </div>
        <Button
          onClick={onRun}
          disabled={isRunning}
          className="w-full bg-stone-900 text-white hover:bg-stone-800 rounded-xl h-11 font-medium"
        >
          {isRunning ? "Running..." : "Run Simulation"}
        </Button>
      </CardContent>
    </Card>
  );
}
