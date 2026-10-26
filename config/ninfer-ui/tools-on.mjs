// Turns on the web UI's global Tools switch (Settings → Tools, off on a fresh install) from the shell,
// for a box where nobody has signed in as admin yet. The UI reads the switch on every chat request, so
// no restart is needed. Built-in tools (current_time, calculator) are on unless switched off; MCP
// servers' tools start off and are switched on in the UI.
//   /opt/ninfer-ui/runtime/bin/node tools-on.mjs [/var/lib/private/ninfer-ui/ui.db]
//   chown ninfer-ui:ninfer-ui /var/lib/private/ninfer-ui/ui.db*   # while the service runs (DynamicUser)
import {DatabaseSync} from "node:sqlite";
const db = new DatabaseSync(process.argv[2] ?? "/var/lib/private/ninfer-ui/ui.db");
db.exec("PRAGMA busy_timeout=5000");
db.prepare("INSERT INTO settings(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value").run("tools_enabled", "true");
console.log("settings:", JSON.stringify(db.prepare("SELECT * FROM settings").all()));
console.log("tool switches:", JSON.stringify(db.prepare("SELECT * FROM tool_settings").all()));
db.close();
