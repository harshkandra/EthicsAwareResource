import { Fragment } from "react";
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
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

export interface AgentRoundData {
  id: number;
  request: number;
  allocation: number;
  waiting: number;
  fulfilled: number;
  bid: number;
  demand: number;
  priority: number;
  ethicsScore: number;
  reasons: string[];
}

interface RoundCardProps {
  roundNumber: number;
  agents: AgentRoundData[];
  isLatest: boolean;
}

export function RoundCard({ roundNumber, agents, isLatest }: RoundCardProps) {
  return (
    <Collapsible defaultOpen>
      <Card className="border-stone-200 bg-white rounded-2xl shadow-sm">
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer bg-amber-50 pb-3 flex-row items-center justify-between rounded-t-2xl">
            <div>
              <CardTitle className="text-stone-800 text-base">Round {roundNumber}</CardTitle>
              <p className="text-xs text-stone-500">Tap to expand or collapse</p>
            </div>
            <div className="flex items-center gap-2">
              {isLatest && (
                <Badge className="bg-amber-100 text-amber-700 border-amber-200 hover:bg-amber-100 text-xs">
                  Latest
                </Badge>
              )}
              <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-stone-100 text-stone-700 text-xs font-semibold">
                ▼
              </span>
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        <CollapsibleContent>
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
                  <Fragment key={agent.id}>
                    <TableRow className="border-stone-50 hover:bg-stone-50/50 transition-colors">
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
                    <TableRow className="bg-stone-50">
                      <TableCell colSpan={6} className="px-6 pb-3 pt-0">
                        <div className="text-xs text-stone-600">
                          <p className="mb-2 font-semibold text-stone-700">Reasoning</p>
                          <ul className="list-disc pl-5 space-y-1">
                            {agent.reasons.map((reason) => (
                              <li key={reason}>{reason}</li>
                            ))}
                          </ul>
                        </div>
                      </TableCell>
                    </TableRow>
                  </Fragment>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
