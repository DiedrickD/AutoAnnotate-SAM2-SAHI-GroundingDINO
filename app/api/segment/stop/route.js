import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const JOB_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export async function POST(request) {
  const { jobId } = await request.json().catch(() => ({}));

  if (!JOB_ID_PATTERN.test(String(jobId || ""))) {
    return Response.json({ error: "Invalid job id." }, { status: 400 });
  }

  const uploadDir = path.join(process.cwd(), "webapp-data", "uploads", jobId);
  await mkdir(uploadDir, { recursive: true });
  await writeFile(path.join(uploadDir, "stop.requested"), new Date().toISOString(), "utf8");

  return Response.json({ ok: true });
}
