import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export interface AgentRoundData {
  id: number;
  request: number;
  allocation: number;
  waiting: number;
  fulfilled: number;
  bid: number;
}

interface RoundCardProps {
  roundNumber: number;
  agents: AgentRoundData[];
  isLatest: boolean;
}

export function RoundCard({ roundNumber, agents, isLatest }: RoundCardProps) {
  return (
    <Card className="border-stone-200 bg-white rounded-2xl shadow-sm">
      <CardHeader className="pb-3 flex-row items-center justify-between">
        <CardTitle className="text-stone-800 text-base">
          Round {roundNumber}
        </CardTitle>
        {isLatest && (
          <Badge className="bg-amber-100 text-amber-700 border-amber-200 hover:bg-amber-100 text-xs">
            Latest
          </Badge>
        )}
      </CardHeader>
      <CardContent className="p-0">
        <Table>
          <TableHeader>
            <TableRow className="border-stone-100 hover:bg-transparent">
              <TableHead className="text-stone-500 text-xs font-medium pl-6">Agent</TableHead>
              <TableHead className="text-stone-500 text-xs font-medium">Req</TableHead>
              <TableHead className="text-stone-500 text-xs font-medium">Alloc</TableHead>
              <TableHead className="text-stone-500 text-xs font-medium">Waiting</TableHead>
              <TableHead className="text-stone-500 text-xs font-medium">Fulfilled</TableHead>
              <TableHead className="text-stone-500 text-xs font-medium pr-6">Bid</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {agents.map((agent) => (
              <TableRow
                key={agent.id}
                className="border-stone-50 hover:bg-stone-50/50 transition-colors"
              >
                <TableCell className="pl-6">
                  <div className="flex items-center gap-2">
                    <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-amber-100 text-amber-700 text-xs font-semibold">
                      {agent.id}
                    </span>
                    <span className="text-sm text-stone-700 font-medium">
                      Agent {agent.id}
                    </span>
                  </div>
                </TableCell>
                <TableCell className="text-sm text-stone-600">{agent.request}</TableCell>
                <TableCell>
                  <span
                    className={`text-sm font-medium ${
                      agent.allocation > 0 ? "text-emerald-600" : "text-red-500"
                    }`}
                  >
                    {agent.allocation}
                  </span>
                </TableCell>
                <TableCell className="text-sm text-stone-600">{agent.waiting}</TableCell>
                <TableCell className="text-sm text-stone-600">{agent.fulfilled}</TableCell>
                <TableCell className="pr-6">
                  <span className="text-sm font-mono text-blue-600">
                    {agent.bid.toFixed(2)}
                  </span>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
