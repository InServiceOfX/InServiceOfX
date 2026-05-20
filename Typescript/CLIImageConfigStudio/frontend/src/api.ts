import type {
  CliImageConfig,
  CliImageStatus,
  ConfigLocation,
  ProfileSummary
} from "./types";

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error ?? `Request failed: ${response.status}`);
  }
  return payload as T;
}

export function loadConfig(): Promise<CliImageConfig> {
  return requestJson<CliImageConfig>("/api/config");
}

export function loadStatus(): Promise<CliImageStatus> {
  return requestJson<CliImageStatus>("/api/status");
}

export function saveConfig(config: Partial<CliImageConfig>): Promise<{ ok: boolean }> {
  return requestJson<{ ok: boolean }>("/api/config", {
    method: "POST",
    body: JSON.stringify(config)
  });
}

export function saveProfile(name: string): Promise<{ name: string; files: string[] }> {
  return requestJson<{ name: string; files: string[] }>("/api/profiles/save", {
    method: "POST",
    body: JSON.stringify({ name })
  });
}

export function applyProfile(
  name: string
): Promise<{ name: string; files: string[]; backup: string | null }> {
  return requestJson<{ name: string; files: string[]; backup: string | null }>(
    "/api/profiles/apply",
    {
      method: "POST",
      body: JSON.stringify({ name })
    }
  );
}

export function loadProfiles(): Promise<{ profiles: ProfileSummary[] }> {
  return requestJson<{ profiles: ProfileSummary[] }>("/api/profiles");
}

export function loadLocation(): Promise<ConfigLocation> {
  return requestJson<ConfigLocation>("/api/location");
}

export function setLocation(
  configDir: string,
  initializeFromExamples: boolean
): Promise<ConfigLocation> {
  return requestJson<ConfigLocation>("/api/location", {
    method: "POST",
    body: JSON.stringify({
      config_dir: configDir,
      initialize_from_examples: initializeFromExamples
    })
  });
}
