"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Signal } from "@/lib/types";
import { OpportunityTable } from "@/components/OpportunityTable";
import { Panel, Select } from "@/components/ui";

export default function HistoryPage() {
  const [rows, setRows] = useState<Signal[]>([]);
  const [status, setStatus] = useState("");

  useEffect(() => {
    api.history(status || undefined).then(setRows).catch(() => undefined);
    const t = setInterval(() => api.history(status || undefined).then(setRows).catch(() => undefined), 10000);
    return () => clearInterval(t);
  }, [status]);

  return (
    <div className="p-4 space-y-4">
      <header className="flex items-end justify-between">
        <div>
          <div className="text-[11px] uppercase tracking-[0.18em] text-mute">Ledger</div>
          <h1 className="text-xl font-semibold">Signal history</h1>
        </div>
        <Select
          label="Status"
          value={status}
          onChange={setStatus}
          options={[
            { value: "", label: "All" },
            { value: "active", label: "Active" },
            { value: "target_1_hit", label: "Target 1" },
            { value: "target_2_hit", label: "Target 2" },
            { value: "stopped", label: "Stopped" },
            { value: "invalidated", label: "Invalidated" },
            { value: "expired", label: "Expired" },
          ]}
        />
      </header>
      <Panel title="Active + historical outcomes">
        <OpportunityTable rows={rows} />
      </Panel>
    </div>
  );
}
