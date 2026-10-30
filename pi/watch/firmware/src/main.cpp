// T-Watch S3: Start/Stop for the court recorder over Bluetooth LE.
//
// The Pi side is pi/watch/bridge.py, which describes the two characteristics.
// Bluetooth runs in its own task (scanning and connecting block); the UI runs
// in loop() and only ever reads what that task leaves in `ble`.

#include <LilyGoLib.h>
#include <LV_Helper.h>
#include <NimBLEDevice.h>
#include <ArduinoJson.h>
#include <string>

static const char *SERVICE_UUID = "a5c0de00-7e9b-4c39-9a35-0c0a0f7e1a00";
static const char *COMMAND_UUID = "a5c0de01-7e9b-4c39-9a35-0c0a0f7e1a00";
static const char *STATUS_UUID  = "a5c0de02-7e9b-4c39-9a35-0c0a0f7e1a00";

static const uint32_t HOLD_MS = 1000;             // press this long to start/stop
static const uint32_t SCREEN_OFF_MS = 20000;
static const uint32_t SLEEP_AFTER_OFF_MS = 60000; // then deep sleep; the crown wakes it
static const uint32_t PENDING_MS = 15000;         // a command's answer, at most
static const uint8_t  BRIGHTNESS = 120;

static const lv_color_t LIME = lv_color_hex(0xdfff00);
static const lv_color_t RED  = lv_color_hex(0xff4d4d);
static const lv_color_t GREY = lv_color_hex(0x454845);
static const lv_color_t DARK = lv_color_hex(0x070807);
static const lv_color_t MUTED = lv_color_hex(0x9ea09e);

// -- shared between the Bluetooth task and the UI ------------------------------

enum Phase { SEARCHING, CONNECTING, PAIRING, CONNECTED };

static struct {
    Phase phase = SEARCHING;
    bool notPaired = false;      // the last pairing was refused; cleared once connected
    bool fresh = false;          // `status` is new
    std::string status;          // the status characteristic's JSON
    std::string command;         // to write ("" = nothing)
    bool sendFailed = false;
} ble;
static SemaphoreHandle_t mtx;
static volatile bool wantLink = true;
static uint32_t passkey;

struct Lock {
    Lock() { xSemaphoreTake(mtx, portMAX_DELAY); }
    ~Lock() { xSemaphoreGive(mtx); }
};

static void setPhase(Phase p)
{
    Lock l;
    ble.phase = p;
}

static void onNotify(NimBLERemoteCharacteristic *, uint8_t *data, size_t len, bool)
{
    Lock l;
    ble.status.assign((const char *)data, len);
    ble.fresh = true;
}

// Find the Pi, connect, make the link authenticated (pairing the first time:
// the Pi types the passkey shown here), subscribe, then pass commands on.
static void bleTask(void *)
{
    NimBLEClient *client = NimBLEDevice::createClient();
    client->setConnectTimeout(5);
    NimBLERemoteCharacteristic *cmd = nullptr;
    for (;;) {
        if (!wantLink) {
            if (client->isConnected()) client->disconnect();
            cmd = nullptr;
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }
        if (!client->isConnected()) {
            cmd = nullptr;
            setPhase(SEARCHING);
            NimBLEScan *scan = NimBLEDevice::getScan();
            NimBLEScanResults found = scan->start(3, false);
            NimBLEAddress pi;
            bool seen = false;
            for (int i = 0; i < found.getCount() && !seen; i++) {
                NimBLEAdvertisedDevice d = found.getDevice(i);
                if (d.isAdvertisingService(NimBLEUUID(SERVICE_UUID))) {
                    pi = d.getAddress();
                    seen = true;
                }
            }
            scan->clearResults();
            if (!seen) {
                vTaskDelay(pdMS_TO_TICKS(1000));
                continue;
            }
            setPhase(CONNECTING);
            if (!client->connect(pi)) {
                vTaskDelay(pdMS_TO_TICKS(1000));
                continue;
            }
            bool bonded = NimBLEDevice::isBonded(pi);
            if (!bonded) setPhase(PAIRING);
            bool ok = client->secureConnection() && client->getConnInfo().isAuthenticated();
            if (!ok) {
                client->disconnect();
                // A bond the Pi no longer has: drop ours and pair afresh.
                if (bonded) NimBLEDevice::deleteBond(pi);
                {
                    Lock l;
                    ble.notPaired = true;
                }
                vTaskDelay(pdMS_TO_TICKS(bonded ? 500 : 8000));
                continue;
            }
            NimBLERemoteService *svc = client->getService(SERVICE_UUID);
            NimBLERemoteCharacteristic *st = svc ? svc->getCharacteristic(STATUS_UUID) : nullptr;
            cmd = svc ? svc->getCharacteristic(COMMAND_UUID) : nullptr;
            if (!st || !cmd || !st->subscribe(true, onNotify)) {
                client->disconnect();
                vTaskDelay(pdMS_TO_TICKS(1000));
                continue;
            }
            std::string first = st->readValue();
            {
                Lock l;
                ble.status = first;
                ble.fresh = true;
                ble.notPaired = false;
                ble.phase = CONNECTED;
            }
        }
        std::string c;
        {
            Lock l;
            c.swap(ble.command);
        }
        if (!c.empty() && cmd && !cmd->writeValue((const uint8_t *)c.data(), c.size(), true)) {
            Lock l;
            ble.sendFailed = true;
        }
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

// -- UI -----------------------------------------------------------------------

static struct {
    std::string s = "offline";   // idle | recording | finishing | offline
    long t = 0;                  // seconds recorded when `at` was set
    uint32_t at = 0;
    std::string err;
    std::string busy;            // "ffmpeg 1234\n..." (empty: camera free)
} rec;

static lv_obj_t *lblLink, *lblTime, *btn, *lblBtn, *lblInfo, *shield, *mbox;
static uint32_t pendingAt = 0;   // a command was sent and not yet answered
static bool askOnBusy = false;   // a "start" was sent: a busy answer opens the prompt
static bool screenOn = true;
static uint32_t offAt = 0;
static volatile bool pmuIrq = false;

static void send(const char *c)
{
    {
        Lock l;
        ble.command = c;
    }
    pendingAt = millis();
    askOnBusy = strcmp(c, "start") == 0;
}

static void onHold(lv_event_t *)
{
    if (pendingAt) return;
    if (rec.s == "idle") send("start");
    else if (rec.s == "recording") send("stop");
}

static void onBusyAnswer(lv_event_t *e)
{
    lv_obj_t *m = lv_event_get_current_target(e);
    const char *b = lv_msgbox_get_active_btn_text(m);
    if (b && strcmp(b, "Close it") == 0) send("start-force");
    lv_msgbox_close(m);
    mbox = nullptr;
}

static void showBusy()
{
    static const char *btns[] = {"Close it", "Cancel", ""};
    std::string text = "Another program has the camera:\n" + rec.busy +
                       "\nClose it and start recording?";
    mbox = lv_msgbox_create(NULL, "Camera in use", text.c_str(), btns, false);
    lv_obj_set_width(mbox, 220);
    lv_obj_center(mbox);
    lv_obj_add_event_cb(mbox, onBusyAnswer, LV_EVENT_VALUE_CHANGED, NULL);
}

static void screenOff()
{
    screenOn = false;
    offAt = millis();
    watch.setBrightness(0);
    lv_obj_clear_flag(shield, LV_OBJ_FLAG_HIDDEN);
}

static void screenWake()
{
    screenOn = true;
    lv_obj_add_flag(shield, LV_OBJ_FLAG_HIDDEN);
    lv_disp_trig_activity(NULL);
    watch.setBrightness(BRIGHTNESS);
}

// While the screen is off, a full-screen layer takes the first touch, so
// waking the watch can never press Start or Stop.
static void onShield(lv_event_t *)
{
    screenWake();
}

static void deepSleep()
{
    wantLink = false;
    for (int i = 0; i < 20 && NimBLEDevice::getClientListSize() &&
                    NimBLEDevice::getClientList()->front()->isConnected(); i++)
        delay(50);
    watch.clearPMU();
    watch.setSleepMode(PMU_BTN_WAKEUP);
    watch.sleep();                     // wakes by rebooting
}

static void buildUi()
{
    lv_obj_t *scr = lv_scr_act();
    lv_obj_set_style_bg_color(scr, DARK, 0);
    lv_obj_set_style_text_color(scr, lv_color_white(), 0);

    lblLink = lv_label_create(scr);
    lv_obj_set_style_text_font(lblLink, &lv_font_montserrat_16, 0);
    lv_obj_set_style_text_color(lblLink, MUTED, 0);
    lv_obj_align(lblLink, LV_ALIGN_TOP_MID, 0, 6);

    lblTime = lv_label_create(scr);
    lv_obj_set_style_text_font(lblTime, &lv_font_montserrat_28, 0);
    lv_obj_align(lblTime, LV_ALIGN_TOP_MID, 0, 28);

    btn = lv_btn_create(scr);
    lv_obj_set_size(btn, 130, 130);
    lv_obj_set_style_radius(btn, LV_RADIUS_CIRCLE, 0);
    lv_obj_set_style_shadow_width(btn, 0, 0);
    lv_obj_align(btn, LV_ALIGN_CENTER, 0, 14);
    lv_obj_add_event_cb(btn, onHold, LV_EVENT_LONG_PRESSED, NULL);
    lblBtn = lv_label_create(btn);
    lv_obj_set_style_text_font(lblBtn, &lv_font_montserrat_26, 0);
    lv_obj_center(lblBtn);

    lblInfo = lv_label_create(scr);
    lv_obj_set_width(lblInfo, 224);
    lv_obj_set_style_text_align(lblInfo, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_set_style_text_font(lblInfo, &lv_font_montserrat_14, 0);
    lv_label_set_long_mode(lblInfo, LV_LABEL_LONG_DOT);
    lv_obj_set_height(lblInfo, 36);
    lv_obj_align(lblInfo, LV_ALIGN_BOTTOM_MID, 0, -2);

    shield = lv_obj_create(lv_layer_top());
    lv_obj_remove_style_all(shield);
    lv_obj_set_size(shield, LV_PCT(100), LV_PCT(100));
    lv_obj_add_flag(shield, LV_OBJ_FLAG_CLICKABLE | LV_OBJ_FLAG_HIDDEN);
    lv_obj_add_event_cb(shield, onShield, LV_EVENT_PRESSED, NULL);

    lv_indev_t *indev = lv_indev_get_next(NULL);
    if (indev) indev->driver->long_press_time = HOLD_MS;
}

static void styleBtn(lv_color_t c, const char *text, bool enabled)
{
    lv_obj_set_style_bg_color(btn, c, 0);
    lv_obj_set_style_text_color(lblBtn, enabled ? DARK : MUTED, 0);
    lv_label_set_text(lblBtn, text);
    if (enabled) lv_obj_clear_state(btn, LV_STATE_DISABLED);
    else lv_obj_add_state(btn, LV_STATE_DISABLED);
}

static void takeStatus(const std::string &json)
{
    JsonDocument d;
    if (deserializeJson(d, json)) return;
    std::string s = d["s"] | "offline";
    if (s != rec.s) askOnBusy = askOnBusy && s == "idle";
    rec.s = s;
    rec.t = d["t"] | 0L;
    rec.at = millis();
    rec.err = d["err"] | "";
    rec.busy.clear();
    for (JsonVariant b : d["busy"].as<JsonArray>()) {
        if (!rec.busy.empty()) rec.busy += "\n";
        rec.busy += b.as<const char *>();
    }
    pendingAt = 0;
    if (!rec.busy.empty() && askOnBusy && !mbox) {
        askOnBusy = false;
        if (!screenOn) screenWake();
        showBusy();
    }
}

static void render(Phase phase, bool notPaired)
{
    char buf[64];
    bool up = phase == CONNECTED;
    if (up) lv_label_set_text(lblLink, LV_SYMBOL_BLUETOOTH " Court Pi");
    else if (phase == PAIRING) lv_label_set_text(lblLink, "Pairing...");
    else if (notPaired) lv_label_set_text(lblLink, "Not paired");
    else if (phase == CONNECTING) lv_label_set_text(lblLink, "Connecting...");
    else lv_label_set_text(lblLink, "Looking for the Pi...");

    if (!up) {
        lv_label_set_text(lblTime, "");
        styleBtn(GREY, LV_SYMBOL_BLUETOOTH, false);
        if (phase == PAIRING || notPaired) {
            snprintf(buf, sizeof buf, "%s%03lu %03lu",
                     phase == PAIRING ? "Type on the Pi: " : "On the Pi run: bridge.py pair\nthen type ",
                     (unsigned long)(passkey / 1000), (unsigned long)(passkey % 1000));
            lv_label_set_text(lblInfo, buf);
        } else {
            lv_label_set_text(lblInfo, "");
        }
        return;
    }

    bool waiting = pendingAt != 0;
    if (rec.s == "recording") {
        long secs = rec.t + (long)((millis() - rec.at) / 1000);
        if (secs >= 3600) snprintf(buf, sizeof buf, "%ld:%02ld:%02ld", secs / 3600, secs / 60 % 60, secs % 60);
        else snprintf(buf, sizeof buf, "%02ld:%02ld", secs / 60, secs % 60);
        lv_obj_set_style_text_color(lblTime, RED, 0);
        lv_label_set_text(lblTime, buf);
        styleBtn(RED, waiting ? "..." : "STOP", !waiting);
    } else if (rec.s == "idle") {
        lv_obj_set_style_text_color(lblTime, lv_color_white(), 0);
        lv_label_set_text(lblTime, "Ready");
        styleBtn(LIME, waiting ? "..." : "START", !waiting);
    } else if (rec.s == "finishing") {
        lv_obj_set_style_text_color(lblTime, lv_color_white(), 0);
        lv_label_set_text(lblTime, "Saving");
        styleBtn(GREY, "...", false);
    } else {
        lv_obj_set_style_text_color(lblTime, MUTED, 0);
        lv_label_set_text(lblTime, "Recorder off");
        styleBtn(GREY, "-", false);
    }
    if (!rec.err.empty()) {
        lv_obj_set_style_text_color(lblInfo, RED, 0);
        lv_label_set_text(lblInfo, rec.err.c_str());
    } else {
        lv_obj_set_style_text_color(lblInfo, MUTED, 0);
        lv_label_set_text(lblInfo, rec.s == "idle" ? "Hold to start"
                                 : rec.s == "recording" ? "Hold to stop" : "");
    }
}

static void IRAM_ATTR onPmu()
{
    pmuIrq = true;
}

void setup()
{
    mtx = xSemaphoreCreateMutex();
    watch.begin();
    beginLvglHelper();
    watch.setBrightness(BRIGHTNESS);
    watch.attachPMU(onPmu);
    buildUi();

    passkey = 100000 + esp_random() % 900000;
    NimBLEDevice::init("Anya Watch");
    NimBLEDevice::setSecurityAuth(true, true, true);          // bond, MITM, LE Secure Connections
    NimBLEDevice::setSecurityIOCap(BLE_HS_IO_DISPLAY_ONLY);   // the Pi types our passkey
    NimBLEDevice::setSecurityPasskey(passkey);
    NimBLEDevice::setMTU(247);
    xTaskCreatePinnedToCore(bleTask, "ble", 8192, NULL, 1, NULL, 0);
}

void loop()
{
    Phase phase;
    bool notPaired, fresh, failed;
    std::string status;
    {
        Lock l;
        phase = ble.phase;
        notPaired = ble.notPaired;
        fresh = ble.fresh;
        failed = ble.sendFailed;
        ble.fresh = ble.sendFailed = false;
        if (fresh) status = ble.status;
    }
    if (fresh) takeStatus(status);
    if (failed || (pendingAt && millis() - pendingAt > PENDING_MS)) {
        pendingAt = 0;
        rec.err = "No answer from the Pi";
    }
    if (phase != CONNECTED && pendingAt) {
        pendingAt = 0;
        rec.err = "Lost the Pi";
    }
    render(phase, notPaired);

    if (pmuIrq) {
        pmuIrq = false;
        watch.readPMU();
        if (watch.isPekeyShortPressIrq()) {
            if (screenOn) screenOff();
            else screenWake();
        }
        watch.clearPMU();
    }
    // Keep the passkey on screen while the Pi waits for it.
    if (phase == PAIRING) lv_disp_trig_activity(NULL);
    if (screenOn && lv_disp_get_inactive_time(NULL) > SCREEN_OFF_MS) screenOff();
    if (!screenOn && millis() - offAt > SLEEP_AFTER_OFF_MS) deepSleep();

    lv_task_handler();
    delay(5);
}
