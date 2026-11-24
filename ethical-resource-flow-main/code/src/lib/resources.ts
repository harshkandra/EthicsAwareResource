export type ResourceVector = Record<string, number>;

export const RESOURCE_LABELS: Record<string, string> = {
  cpu: "CPU",
  ram_gb: "RAM GB",
  gpu: "GPU",
  network_mbps: "Network Mbps",
  storage_iops: "Storage IOPS",
};

export function formatResources(resources: ResourceVector): string {
  return Object.entries(RESOURCE_LABELS)
    .map(([key, label]) => `${label} ${resources[key] ?? 0}`)
    .join(" · ");
}
