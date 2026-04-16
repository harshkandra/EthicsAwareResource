import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const definitions = [
  { term: "Waiting", desc: "Number of rounds agent got 0 resource", color: "bg-red-400" },
  { term: "Fulfilled", desc: "Number of times demand fully satisfied", color: "bg-emerald-400" },
  { term: "Bid", desc: "Competitive strength in negotiation", color: "bg-blue-400" },
  { term: "Utility", desc: "Priority-based fairness function", color: "bg-amber-400" },
];

export function DefinitionsPanel() {
  return (
    <div className="space-y-4">
      <Card className="border-stone-200 bg-white rounded-2xl shadow-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-stone-800 text-lg">Definitions</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {definitions.map((d) => (
            <div key={d.term} className="flex items-start gap-3">
              <span className={`mt-1.5 h-2.5 w-2.5 rounded-full ${d.color} shrink-0`} />
              <div>
                <p className="text-sm font-medium text-stone-800">{d.term}</p>
                <p className="text-xs text-stone-500">{d.desc}</p>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card className="border-stone-200 bg-white rounded-2xl shadow-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-stone-800 text-lg">Made By</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1">
          <p className="text-sm text-stone-700">Saurabh Tripathi - M251020CS</p>
          <p className="text-sm text-stone-700">Harsh Kumar Kandra -M250791CS</p>
          <p className="text-xs text-stone-400 mt-1">NIT Calicut</p>
        </CardContent>
      </Card>
    </div>
  );
}
