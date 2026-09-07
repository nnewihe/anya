/**
 * Seed the free-year allowlist, and comp an account by hand.
 *
 *   npx ts-node scripts/seed-grandfathered.ts add a@b.com c@d.com
 *   npx ts-node scripts/seed-grandfathered.ts list
 *   npx ts-node scripts/seed-grandfathered.ts remove a@b.com
 *   npx ts-node scripts/seed-grandfathered.ts comp a@b.com 365
 *
 * Run against the emulator by exporting FIRESTORE_EMULATOR_HOST first; against
 * production it needs GOOGLE_APPLICATION_CREDENTIALS pointing at a
 * service-account key. That key must never be committed — .gitignore matches
 * serviceAccount*.json for exactly this reason.
 *
 * `list` prints digests, not addresses: the collection stores sha256 of the
 * lowercased email and cannot be reversed. Keep the source list of who was in
 * the beta somewhere else; this is a membership test, not a record.
 */
import * as admin from "firebase-admin";
import { createHash } from "crypto";

admin.initializeApp();
const db = admin.firestore();

const hash = (email: string) =>
  createHash("sha256").update(email.trim().toLowerCase(), "utf8").digest("hex");

async function main() {
  const [command, ...args] = process.argv.slice(2);

  switch (command) {
    case "add": {
      if (!args.length) throw new Error("usage: add <email> [email...]");
      const batch = db.batch();
      for (const email of args) {
        batch.set(db.doc(`grandfathered/${hash(email)}`), {
          addedAt: Math.floor(Date.now() / 1000),
          note: "beta tester",
        });
      }
      await batch.commit();
      console.log(`added ${args.length} address(es)`);
      return;
    }

    case "remove": {
      if (!args.length) throw new Error("usage: remove <email> [email...]");
      const batch = db.batch();
      for (const email of args) batch.delete(db.doc(`grandfathered/${hash(email)}`));
      await batch.commit();
      console.log(`removed ${args.length} address(es)`);
      return;
    }

    case "check": {
      for (const email of args) {
        const doc = await db.doc(`grandfathered/${hash(email)}`).get();
        console.log(`${email}: ${doc.exists ? "on the list" : "not on the list"}`);
      }
      return;
    }

    case "list": {
      const snap = await db.collection("grandfathered").get();
      console.log(`${snap.size} entries (digests, not addresses):`);
      snap.forEach((d) => console.log(`  ${d.id}  ${JSON.stringify(d.data())}`));
      return;
    }

    case "comp": {
      // The escape hatch the rollout plan calls for: someone will need
      // comping in the first month, and doing it by hand in the console means
      // getting the claim and the document to agree, which is easy to botch.
      const [email, daysRaw] = args;
      if (!email) throw new Error("usage: comp <email> [days=365]");
      const days = Number(daysRaw ?? 365);
      const user = await admin.auth().getUserByEmail(email);
      const until = Math.floor(Date.now() / 1000) + days * 86400;

      await db.doc(`users/${user.uid}`).set(
        {
          entitlement: { active: true, until, source: "comp" },
          updatedAt: Math.floor(Date.now() / 1000),
        },
        { merge: true }
      );
      await admin.auth().setCustomUserClaims(user.uid, {
        ...(user.customClaims ?? {}),
        ent: 1,
        entExp: until,
      });
      console.log(`comped ${email} (${user.uid}) for ${days} days`);
      console.log("They must sign out and back in, or relaunch, to pick up the claim.");
      return;
    }

    default:
      console.error("commands: add | remove | check | list | comp");
      process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
