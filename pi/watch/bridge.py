#!/usr/bin/env python3
"""
Bluetooth remote for the court recorder: a paired LilyGo T-Watch 2020 presses
Start and Stop.

    python3 bridge.py [--recorder http://127.0.0.1:8080]     the service
    sudo python3 bridge.py pair                               once per watch
    sudo python3 bridge.py list | unpair <address>

The Pi advertises one Bluetooth LE service, and the watch connects to it while
its screen is on.  The service only passes presses on to the recorder's own
web API (pi/recorder/recorder.py, the same calls the page makes), so
recording, the 90-minute limit and the processing afterwards are exactly the
page's, and the page and the watch always show the same state.

Two characteristics:
  command  write   b"start", b"start-force" (close whatever holds the camera
                   first, as the page's OK does), b"stop"
  status   read +  compact JSON, sent on every change and every 10 s:
           notify  {"s": "idle"|"recording"|"finishing"|"offline",
                    "t": seconds recorded, "max": limit in seconds,
                    "saving": recordings still being converted,
                    "err": "...", "busy": ["ffmpeg 1234", ...]}
                   "busy" answers a "start" when another program has the
                   camera; the watch then asks before sending "start-force".

Both need an authenticated, bonded link (BlueZ's encrypt-authenticated
flags), and commands are taken only from a device `pair` marked Trusted.
`pair` makes the Pi pairable for two minutes; the watch shows six digits,
which are typed here.  Pairing without the digits (BLE "Just Works") gives an
unauthenticated link, which the flags refuse.

Needs BlueZ and dbus-next; the service user must be in the bluetooth group.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request

SERVICE_UUID = "a5c0de00-7e9b-4c39-9a35-0c0a0f7e1a00"
COMMAND_UUID = "a5c0de01-7e9b-4c39-9a35-0c0a0f7e1a00"
STATUS_UUID = "a5c0de02-7e9b-4c39-9a35-0c0a0f7e1a00"

# The watch asks for an MTU of 247 (244 bytes of value); stay well under it so
# a default-MTU read (with offsets) also works.
MAX_STATUS = 180
POLL_S = 1.0
RESEND_S = 10.0             # the watch ticks the clock itself in between
PAIR_S = 120

COMMANDS = {
    b"start": ("/api/start", {}),
    b"start-force": ("/api/start", {"force": True}),
    b"stop": ("/api/stop", {}),
}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# -- the recorder's web API (no Bluetooth: testable on a laptop) ---------------

def program_name(command):
    """'/usr/bin/ffmpeg -i udp://...' -> 'ffmpeg'; 'python3 x/fake.py' -> 'fake.py'."""
    words = (command or "").split()
    if not words:
        return "?"
    name = os.path.basename(words[0])
    if name.startswith("python") and len(words) > 1:
        name = os.path.basename(words[1])
    return name


def busy_names(procs):
    """The recorder's 409 `busy` list ([{pid, command}]) -> ["ffmpeg 1234"]."""
    return [f"{program_name(p.get('command'))} {p.get('pid')}" for p in procs or []]


def command_to_request(data):
    """The bytes written to the command characteristic -> (path, JSON body),
    or None for anything else."""
    return COMMANDS.get(bytes(data).strip().lower())


def encode_status(st, err=None, busy=None):
    """The recorder's /api/status (None: unreachable), plus the outcome of the
    last command -> the status characteristic's bytes."""
    if st is None:
        d = {"s": "offline"}
    else:
        d = {"s": st.get("state"),
             "t": int(st.get("elapsed_s") or 0),
             "max": st.get("max_s"),
             "saving": st.get("saving") or 0}
    err = err or (st or {}).get("error")
    if err:
        d["err"] = err
    if busy:
        d["busy"] = list(busy)

    def dump():
        return json.dumps(d, separators=(",", ":"), ensure_ascii=False).encode()

    out = dump()
    while len(out) > MAX_STATUS and len(d.get("busy", [])) > 1:
        d["busy"].pop()
        out = dump()
    while len(out) > MAX_STATUS and len(d.get("err", "")) > 1:
        d["err"] = d["err"][:-4] + "…"
        out = dump()
    return out


class RecorderClient:
    def __init__(self, base="http://127.0.0.1:8080", timeout=20):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def status(self):
        """/api/status, or None when the recorder doesn't answer."""
        try:
            with urllib.request.urlopen(self.base + "/api/status", timeout=5) as r:
                return json.load(r)
        except (OSError, ValueError):
            return None

    def post(self, path, body):
        """-> (status or None, error or None, busy procs or None)."""
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            # Start can take seconds: it asks the GoPro to stream, or closes
            # another program first.
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r), None, None
        except urllib.error.HTTPError as e:
            try:
                b = json.load(e)
            except ValueError:
                b = {}
            return None, b.get("error") or f"recorder said {e.code}", b.get("busy")
        except (OSError, ValueError):
            return None, "The recorder isn't answering.", None


class Remote:
    """What the status characteristic says, kept from the recorder's state and
    the last command.  `notify(bytes)` is called when the watch should hear."""

    def __init__(self, client, notify=lambda value: None, clock=time.monotonic):
        self.client = client
        self.notify = notify
        self.clock = clock
        self.st = None
        self.err = None              # the last command's error, until the next
        self.busy = None             # ... or until the state changes
        self.value = encode_status(None)
        self._key = None
        self._sent = float("-inf")
        self._cmd_lock = asyncio.Lock()

    def _publish(self, force=False):
        self.value = encode_status(self.st, self.err, self.busy)
        d = json.loads(self.value)
        d.pop("t", None)
        key = json.dumps(d, sort_keys=True)
        now = self.clock()
        if force or key != self._key or now - self._sent >= RESEND_S:
            self._key, self._sent = key, now
            self.notify(self.value)

    def update(self, st):
        """A fresh /api/status (None: unreachable)."""
        old = (self.st or {}).get("state")
        self.st = st
        if (st or {}).get("state") != old:
            self.err = self.busy = None
        self._publish()

    async def poll(self):
        self.update(await asyncio.to_thread(self.client.status))

    async def command(self, data):
        req = command_to_request(data)
        if req is None:
            self.err, self.busy = f"unknown command {bytes(data)[:20]!r}", None
            return self._publish(force=True)
        async with self._cmd_lock:
            st, err, busy = await asyncio.to_thread(self.client.post, *req)
        log(f"{req[0]} {req[1] or ''} -> {err or 'ok'}")
        self.err, self.busy = err, busy_names(busy) if busy else None
        if st is not None:
            self.st = st
        self._publish(force=True)


# -- BlueZ over D-Bus ----------------------------------------------------------

try:
    from dbus_next import BusType, DBusError, Message, MessageType, Variant
    from dbus_next.aio import MessageBus
    from dbus_next.constants import PropertyAccess
    from dbus_next.service import ServiceInterface, dbus_property, method
except ImportError:                         # the tests above need none of it
    ServiceInterface = None

BLUEZ = "org.bluez"
APP_PATH = "/org/anya/watch"
AGENT_PATH = "/org/anya/watch/agent"


async def _iface(bus, path, name):
    obj = bus.get_proxy_object(BLUEZ, path, await bus.introspect(BLUEZ, path))
    return obj.get_interface(name)


async def find_adapter(bus):
    om = await _iface(bus, "/", "org.freedesktop.DBus.ObjectManager")
    for path, ifaces in (await om.call_get_managed_objects()).items():
        if "org.bluez.GattManager1" in ifaces and "org.bluez.LEAdvertisingManager1" in ifaces:
            return path
    raise SystemExit("no Bluetooth LE adapter found (is bluetoothd running?)")


async def devices(bus, adapter):
    """{path: Device1 properties} for the devices BlueZ knows on `adapter`."""
    om = await _iface(bus, "/", "org.freedesktop.DBus.ObjectManager")
    out = {}
    for path, ifaces in (await om.call_get_managed_objects()).items():
        dev = ifaces.get("org.bluez.Device1")
        if dev and path.startswith(adapter + "/"):
            out[path] = {k: v.value for k, v in dev.items()}
    return out


async def set_prop(bus, path, iface, name, sig, value):
    props = await _iface(bus, path, "org.freedesktop.DBus.Properties")
    await props.call_set(iface, name, Variant(sig, value))


if ServiceInterface:
    class GattService(ServiceInterface):
        def __init__(self, path):
            super().__init__("org.bluez.GattService1")
            self.path = path

        @dbus_property(access=PropertyAccess.READ)
        def UUID(self) -> "s":
            return SERVICE_UUID

        @dbus_property(access=PropertyAccess.READ)
        def Primary(self) -> "b":
            return True

    class Characteristic(ServiceInterface):
        def __init__(self, path, service, uuid, flags):
            super().__init__("org.bluez.GattCharacteristic1")
            self.path, self.service, self.uuid, self.flags = path, service, uuid, flags
            self.value = b""

        @dbus_property(access=PropertyAccess.READ)
        def UUID(self) -> "s":
            return self.uuid

        @dbus_property(access=PropertyAccess.READ)
        def Service(self) -> "o":
            return self.service

        @dbus_property(access=PropertyAccess.READ)
        def Flags(self) -> "as":
            return self.flags

        @dbus_property(access=PropertyAccess.READ)
        def Value(self) -> "ay":
            return self.value

    class StatusChar(Characteristic):
        def __init__(self, path, service, remote):
            super().__init__(path, service, STATUS_UUID,
                             ["read", "notify", "encrypt-authenticated-read"])
            self.remote = remote
            self.notifying = False

        def send(self, value):
            self.value = value
            if self.notifying:
                self.emit_properties_changed({"Value": value})

        @method()
        def ReadValue(self, options: "a{sv}") -> "ay":
            off = options["offset"].value if "offset" in options else 0
            return self.remote.value[off:]

        @method()
        def StartNotify(self):
            self.notifying = True
            self.send(self.remote.value)

        @method()
        def StopNotify(self):
            self.notifying = False

    class CommandChar(Characteristic):
        def __init__(self, path, service, remote, bus):
            super().__init__(path, service, COMMAND_UUID,
                             ["write", "encrypt-authenticated-write"])
            self.remote, self.bus = remote, bus

        @method()
        async def WriteValue(self, value: "ay", options: "a{sv}"):
            dev = options["device"].value if "device" in options else None
            trusted = False
            if dev:
                try:
                    props = await _iface(self.bus, dev, "org.freedesktop.DBus.Properties")
                    trusted = (await props.call_get("org.bluez.Device1", "Trusted")).value
                except DBusError:
                    pass
            if not trusted:
                log(f"refused {bytes(value)[:20]!r} from {dev}: not paired with `bridge.py pair`")
                raise DBusError("org.bluez.Error.NotAuthorized", "not a paired watch")
            # Answer the write at once; the outcome comes back as a status.
            asyncio.ensure_future(self.remote.command(bytes(value)))

    class Advertisement(ServiceInterface):
        # 31 bytes hold the flags and one 128-bit UUID, not a name as well.
        def __init__(self):
            super().__init__("org.bluez.LEAdvertisement1")

        @dbus_property(access=PropertyAccess.READ)
        def Type(self) -> "s":
            return "peripheral"

        @dbus_property(access=PropertyAccess.READ)
        def ServiceUUIDs(self) -> "as":
            return [SERVICE_UUID]

        @dbus_property(access=PropertyAccess.READ)
        def Discoverable(self) -> "b":
            return True

        @method()
        def Release(self):
            pass

    class PairingAgent(ServiceInterface):
        """Passkey entry: the watch shows six digits, they're typed here."""

        def __init__(self):
            super().__init__("org.bluez.Agent1")

        @method()
        def Release(self):
            pass

        @method()
        async def RequestPasskey(self, device: "o") -> "u":
            line = await asyncio.to_thread(
                input, f"\n{device.rsplit('/', 1)[-1]} wants to pair. "
                       "Type the 6 digits shown on the watch: ")
            digits = "".join(c for c in line if c.isdigit())
            if len(digits) != 6:
                raise DBusError("org.bluez.Error.Rejected", "need 6 digits")
            return int(digits)

        @method()
        async def RequestConfirmation(self, device: "o", passkey: "u"):
            # Numeric comparison: only if the watch also has a yes/no.
            line = await asyncio.to_thread(input, f"\nDoes the watch show {passkey:06d}? [y/N] ")
            if line.strip().lower() != "y":
                raise DBusError("org.bluez.Error.Rejected", "not confirmed")

        @method()
        def RequestAuthorization(self, device: "o"):
            # "Just Works": no digits, so no protection.  Refuse.
            raise DBusError("org.bluez.Error.Rejected", "passkey pairing only")

        @method()
        def AuthorizeService(self, device: "o", uuid: "s"):
            pass

        @method()
        def Cancel(self):
            print("\npairing cancelled by the other side", flush=True)


async def _add_match(bus, rule):
    await bus.call(Message(destination="org.freedesktop.DBus", path="/org/freedesktop/DBus",
                           interface="org.freedesktop.DBus", member="AddMatch",
                           signature="s", body=[rule]))


async def serve(base):
    bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
    adapter = await find_adapter(bus)
    await set_prop(bus, adapter, "org.bluez.Adapter1", "Powered", "b", True)
    try:   # pairing only while `bridge.py pair` runs
        await set_prop(bus, adapter, "org.bluez.Adapter1", "Pairable", "b", False)
    except DBusError as e:
        log(f"couldn't make the adapter unpairable: {e}")

    remote = Remote(RecorderClient(base))
    svc_path = APP_PATH + "/service0"
    status = StatusChar(svc_path + "/char0", svc_path, remote)
    command = CommandChar(svc_path + "/char1", svc_path, remote, bus)
    remote.notify = status.send
    bus.export(svc_path, GattService(svc_path))
    bus.export(status.path, status)
    bus.export(command.path, command)
    advert = Advertisement()
    bus.export(APP_PATH + "/advert0", advert)

    gatt = await _iface(bus, adapter, "org.bluez.GattManager1")
    ads = await _iface(bus, adapter, "org.bluez.LEAdvertisingManager1")
    await gatt.call_register_application(APP_PATH, {})

    async def advertise():
        try:
            await ads.call_unregister_advertisement(APP_PATH + "/advert0")
        except DBusError:
            pass
        await ads.call_register_advertisement(APP_PATH + "/advert0", {})

    await advertise()
    log(f"advertising {SERVICE_UUID} on {adapter}; recorder at {base}")

    # Log connections, and advertise again after a disconnect in case the
    # controller didn't resume it by itself.
    def on_signal(msg):
        if (msg.message_type == MessageType.SIGNAL and msg.member == "PropertiesChanged"
                and msg.path and msg.path.startswith(adapter + "/dev_")):
            iface, changed, _ = msg.body
            if iface == "org.bluez.Device1" and "Connected" in changed:
                up = changed["Connected"].value
                log(f"{msg.path.rsplit('/', 1)[-1]} {'connected' if up else 'disconnected'}")
                if not up:
                    asyncio.ensure_future(advertise())
    bus.add_message_handler(on_signal)
    await _add_match(bus, "type='signal',sender='org.bluez',"
                          "interface='org.freedesktop.DBus.Properties',member='PropertiesChanged'")

    was = None
    while True:
        await remote.poll()
        now = (remote.st or {}).get("state", "offline")
        if now != was:
            log(f"recorder: {now}")
            was = now
        await asyncio.sleep(POLL_S)


async def pair():
    bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
    adapter = await find_adapter(bus)
    bus.export(AGENT_PATH, PairingAgent())
    agents = await _iface(bus, "/org/bluez", "org.bluez.AgentManager1")
    await agents.call_register_agent(AGENT_PATH, "KeyboardOnly")
    await agents.call_request_default_agent(AGENT_PATH)
    before = {p for p, d in (await devices(bus, adapter)).items() if d.get("Paired")}
    A = "org.bluez.Adapter1"
    await set_prop(bus, adapter, A, "Powered", "b", True)
    await set_prop(bus, adapter, A, "PairableTimeout", "u", PAIR_S)
    await set_prop(bus, adapter, A, "Pairable", "b", True)
    print(f"Pairable for {PAIR_S} s.  Wake the watch near the Pi: an unpaired watch "
          "connects and shows six digits.", flush=True)
    try:
        end = time.monotonic() + PAIR_S
        while time.monotonic() < end:
            for path, d in (await devices(bus, adapter)).items():
                if d.get("Paired") and path not in before:
                    await set_prop(bus, path, "org.bluez.Device1", "Trusted", "b", True)
                    print(f"\nPaired and trusted {d.get('Address')} ({d.get('Name') or d.get('Alias')}).")
                    return 0
            await asyncio.sleep(0.5)
        print("\nNo watch paired in time.")
        return 1
    finally:
        try:
            await set_prop(bus, adapter, A, "Pairable", "b", False)
            await agents.call_unregister_agent(AGENT_PATH)
        except DBusError:
            pass


async def list_devices():
    bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
    adapter = await find_adapter(bus)
    rows = [d for d in (await devices(bus, adapter)).values() if d.get("Paired")]
    for d in rows:
        print(f"{d.get('Address')}  {'trusted' if d.get('Trusted') else 'NOT trusted (commands refused)'}"
              f"  {'connected' if d.get('Connected') else ''}  {d.get('Name') or d.get('Alias') or ''}")
    if not rows:
        print("nothing paired")
    return 0


async def unpair(address):
    bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
    adapter = await find_adapter(bus)
    for path, d in (await devices(bus, adapter)).items():
        if d.get("Address", "").lower() == address.lower():
            await (await _iface(bus, adapter, "org.bluez.Adapter1")).call_remove_device(path)
            print(f"removed {address}")
            return 0
    print(f"{address} isn't known here (see `bridge.py list`)")
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--recorder", default="http://127.0.0.1:8080",
                    help="the recorder's web address (default %(default)s)")
    sub.add_parser("serve", help="run the Bluetooth service (the default)")
    sub.add_parser("pair", help="pair a watch: type the digits it shows")
    sub.add_parser("list", help="paired devices")
    un = sub.add_parser("unpair", help="forget a device")
    un.add_argument("address")
    a = ap.parse_args(argv)
    if ServiceInterface is None:
        raise SystemExit("needs dbus-next:  pip install dbus-next")
    if a.cmd == "pair":
        return asyncio.run(pair())
    if a.cmd == "list":
        return asyncio.run(list_devices())
    if a.cmd == "unpair":
        return asyncio.run(unpair(a.address))
    try:
        asyncio.run(serve(a.recorder))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
