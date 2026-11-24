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
import { formatResources, type ResourceVector } from "@/lib/resources";

export interface AgentRoundData {
  id: number;
  name: string;
  objective: string;
  strategy: string;
  requested: ResourceVector;
  allocated: ResourceVector;
  minimum: ResourceVector;
  waiting: number;
  fulfilled: number;
  bid: number;
  fulfillment: number;
  negotiation: string;
  reasons: string[];
}

interface RoundCardProps {
  roundNumber: number;
  agents: AgentRoundData[];
  isLatest: boolean;
  utilization: number;
  fairnessIndex: number;
}

export function RoundCard({
  roundNumber,
  agents,
  isLatest,
  utilization,
  fairnessIndex,
}: RoundCardProps) {
  return (
    <Collapsible defaultOpen>
      <Card className="border-stone-200 bg-white rounded-2xl shadow-sm">
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer bg-amber-50 pb-3 flex-row items-center justify-between rounded-t-2xl">
            <div>
              <CardTitle className="text-stone-800 text-base">Round {roundNumber}</CardTitle>
              <p className="text-xs text-stone-500">
                Utilization {(utilization * 100).toFixed(1)}% · Fairness{" "}
                {(fairnessIndex * 100).toFixed(1)}%
              </p>
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
                  <TableHead className="pl-6 text-xs font-medium text-stone-500">Agent</TableHead>
                  <TableHead className="text-xs font-medium text-stone-500">Request</TableHead>
                  <TableHead className="text-xs font-medium text-stone-500">Allocation</TableHead>
                  <TableHead className="text-xs font-medium text-stone-500">Fulfillment</TableHead>
                  <TableHead className="pr-6 text-xs font-medium text-stone-500">Bid</TableHead>
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
                          <div>
                            <p className="text-sm font-medium text-stone-700">{agent.name}</p>
                            <p className="max-w-56 text-xs text-stone-500">{agent.objective}</p>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <p className="min-w-52 text-xs leading-5 text-stone-600">
                          {formatResources(agent.requested)}
                        </p>
                      </TableCell>
                      <TableCell>
                        <p className="min-w-52 text-xs font-medium leading-5 text-emerald-700">
                          {formatResources(agent.allocated)}
                        </p>
                        <p className="text-xs text-stone-400">
                          Protected min: {formatResources(agent.minimum)}
                        </p>
                      </TableCell>
                      <TableCell className="text-sm text-stone-600">
                        {(agent.fulfillment * 100).toFixed(0)}%
                        <p className="text-xs text-stone-400">
                          Waiting {agent.waiting} · Fulfilled {agent.fulfilled}
                        </p>
                      </TableCell>
                      <TableCell className="pr-6">
                        <span className="text-sm font-mono text-blue-600">
                          {agent.bid.toFixed(2)}
                        </span>
                      </TableCell>
                    </TableRow>
                    <TableRow className="bg-stone-50">
                      <TableCell colSpan={5} className="px-6 pb-3 pt-0">
                        <div className="text-xs text-stone-600">
                          <p className="font-medium text-stone-700">{agent.strategy}</p>
                          <p className="mt-1">{agent.negotiation}</p>
                          <ul className="mt-1 list-disc space-y-1 pl-5">
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
