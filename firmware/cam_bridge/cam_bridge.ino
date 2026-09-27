// ============================================================
// ROVER: wireless hub  (AI-Thinker ESP32-CAM)
// ============================================================
// Owns: the WiFi access point and the MJPEG video server. Nothing else.
//
// The UDP <-> UART bridge that used to live here is GONE, along with the
// wiring to the Nano. The Nano now joins this board's access point as a
// station and speaks UDP straight to the laptop, so this board never
// touches the rover protocol and there are no inter-board wires at all.
// (If you ever need the UART path back, it is in git history.)
//
// Wiring
//   GPIO1/GPIO3 are the FTDI programming header. Nothing else is wired.
//
// On the PC: join WiFi "rover" / "rover1234", then
//   video   http://192.168.4.1/          <- test page in a browser
//           http://192.168.4.1/stream    <- raw MJPEG for OpenCV
//           http://192.168.4.1/health    <- JSON diagnostics, no cable needed
//   data    UDP 192.168.4.50:4210        <- the NANO, not this board
//
// "clients" in /health is the quickest check that the Nano associated:
// expect 2 once both the laptop and the Nano have joined.
// ============================================================

#include <WiFi.h>
#include "esp_camera.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// ---------- config ----------
// This board always hosts its own access point. It never joins another network.
// The Nano and the laptop both associate to it, and the laptop reaches the drone
// on a second WiFi adapter.
//
// (An earlier version could instead join the drone's AP to save needing that
// second adapter. It was dropped: it put the glasses video on the Tello's own
// embedded radio alongside its 720p feed, which inflates exactly the latency
// numbers main.py measures. It is in git history if it is ever needed.)
const char *AP_SSID = "rover";
const char *AP_PASS = "rover1234";      // >= 8 chars, or the AP is left open


#define TARGET_FPS 12                   // frame gate; consumer wants ~10, 12 covers
                                        // the gate's millisecond quantisation
const uint32_t FRAME_MIN_MS = 1000 / TARGET_FPS;

// ---------- AI-Thinker camera pinout ----------
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

WiFiServer httpServer(80);


// /health state, written by the stream loop, read by the /health handler.
volatile bool  camOK      = false;   // did the last frame grab succeed
volatile float measuredFps = 0.0f;   // rolling average over the last window

// ---------- camera ----------
bool cameraBegin() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer   = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;   c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;   c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;   c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;   c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk    = XCLK_GPIO_NUM;
  c.pin_pclk    = PCLK_GPIO_NUM;
  c.pin_vsync   = VSYNC_GPIO_NUM;
  c.pin_href    = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn    = PWDN_GPIO_NUM;
  c.pin_reset   = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    c.frame_size   = FRAMESIZE_VGA;     // 640x480
    c.jpeg_quality = 12;                // lower number = better = bigger
    c.fb_count     = 2;
    c.fb_location  = CAMERA_FB_IN_PSRAM;
    c.grab_mode    = CAMERA_GRAB_LATEST;
  } else {
    c.frame_size   = FRAMESIZE_QVGA;
    c.jpeg_quality = 15;
    c.fb_count     = 1;
    c.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  }

  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    Serial.printf("camera init failed: 0x%x\n", err);
    return false;
  }
  return true;
}

// ---------- network ----------
void netBegin() {
  WiFi.setSleep(false);          // power save adds 100ms+ of RX jitter
  WiFi.mode(WIFI_AP);
  if (!WiFi.softAP(AP_SSID, AP_PASS)) {
    Serial.println("softAP FAILED - nothing can reach this board");
    return;
  }
  Serial.printf("AP up: join \"%s\" then http://%s/\n",
                AP_SSID, WiFi.softAPIP().toString().c_str());
}

// ---------- HTTP / MJPEG (core 0) ----------
static const char *BOUNDARY = "rovframe";

void serveIndex(WiFiClient &c) {
  const char *body =
    "<!doctype html><title>rover</title>"
    "<body style='margin:0;background:#111;color:#eee;font:14px system-ui'>"
    "<p style='padding:8px'>MJPEG: <code>/stream</code> &middot; "
    "diagnostics: <a style='color:#6cf' href='/health'>/health</a></p>"
    "<img src='/stream' style='width:100%;max-width:640px'>"
    "</body>";
  c.printf("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
           "Content-Length: %u\r\nConnection: close\r\n\r\n%s",
           (unsigned)strlen(body), body);
}

// One line of JSON, so a teammate with no serial cable can tell a wedged camera
// apart from a Nano that never associated.
void serveHealth(WiFiClient &c) {
  // This board is always an AP, so there is no upstream to measure RSSI against.
  // The field stays in the JSON only so the shape does not change under the host.
  int rssi = 0;
  // "clients" replaced the old "peer" field: this board no longer handles the
  // data link, so it has no idea where the laptop is. What it CAN report is how
  // many stations are associated - expect 2 once the laptop and the Nano have
  // both joined, which is the fastest way to tell whether the Nano is up.
  int clients = (int)WiFi.softAPgetStationNum();

  char body[192];
  int n = snprintf(body, sizeof(body),
    "{\"cam\":%s,\"fps\":%.1f,\"clients\":%d,\"uptime_ms\":%lu,\"rssi\":%d}",
    camOK ? "true" : "false", measuredFps, clients,
    (unsigned long)millis(), rssi);

  c.printf("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
           "Access-Control-Allow-Origin: *\r\nCache-Control: no-store\r\n"
           "Content-Length: %u\r\nConnection: close\r\n\r\n%s",
           (unsigned)n, body);
}

void serveStream(WiFiClient &c) {
  c.printf("HTTP/1.1 200 OK\r\n"
           "Content-Type: multipart/x-mixed-replace; boundary=%s\r\n"
           "Access-Control-Allow-Origin: *\r\n"
           "Cache-Control: no-store\r\n\r\n", BOUNDARY);

  uint32_t nextFrame = 0;
  int nullStreak = 0;
  uint32_t fpsWindowStart = millis();
  uint32_t fpsFrames = 0;

  while (c.connected()) {
    uint32_t now = millis();
    if (now < nextFrame) { vTaskDelay(pdMS_TO_TICKS(nextFrame - now)); continue; }
    nextFrame = now + FRAME_MIN_MS;

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      // Don't fail silently: a wedged OV2640 looks exactly like a broken
      // <img> in the browser while every other part of the server works.
      if (++nullStreak == 10) {
        Serial.println("camera returned no frames 10x - OV2640 is wedged. "
                       "Pull USB power and replug; RESET alone does not clear it.");
      }
      if (nullStreak >= 40) {
        Serial.println("attempting camera reinit...");
        esp_camera_deinit();
        delay(100);
        Serial.println(cameraBegin() ? "reinit ok" : "reinit failed, power-cycle needed");
        nullStreak = 0;
      }
      camOK = false;
      vTaskDelay(pdMS_TO_TICKS(20));
      continue;
    }
    nullStreak = 0;
    camOK = true;

    // Rolling measured FPS over ~1 s windows. This is the only place that
    // knows what the pipeline is really achieving vs. what TARGET_FPS asks.
    fpsFrames++;
    uint32_t win = millis() - fpsWindowStart;
    if (win >= 1000) {
      measuredFps = (fpsFrames * 1000.0f) / (float)win;
      fpsFrames = 0;
      fpsWindowStart = millis();
    }

    // Cache len before handing the buffer back: reading fb->len after
    // esp_camera_fb_return() is a use-after-return.
    size_t len = fb->len;
    c.printf("--%s\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
             BOUNDARY, (unsigned)len);
    size_t sent = c.write(fb->buf, len);
    c.print("\r\n");
    esp_camera_fb_return(fb);

    if (sent != len) break;   // client went away mid-frame
  }
}

void httpTask(void *) {
  httpServer.begin();
  httpServer.setNoDelay(true);

  for (;;) {
    WiFiClient c = httpServer.accept();
    if (!c) { vTaskDelay(pdMS_TO_TICKS(5)); continue; }

    // Read the request line, discard the rest of the headers.
    // MILLISECONDS: WiFiClient inherits Stream::setTimeout, which is ms. (Only
    // WiFiServer::setTimeout takes seconds.) accept() returns as soon as the TCP
    // handshake completes, one RTT BEFORE the request line arrives, so a 2 ms
    // timeout returned an empty reqLine and routed /stream to the index page.
    c.setTimeout(2000);
    String reqLine = c.readStringUntil('\n');
    while (c.available()) {
      String h = c.readStringUntil('\n');
      if (h.length() <= 1) break;
    }

    if      (reqLine.indexOf("/stream") >= 0) serveStream(c);
    else if (reqLine.indexOf("/health") >= 0) serveHealth(c);
    else                                      serveIndex(c);

    c.stop();
  }
}

// ---------- main ----------
void setup() {
  // ESP32-CAM browns out easily with camera + AP on a weak supply.
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);                                   // FTDI debug, GPIO1/3
  Serial.println();

  if (!cameraBegin()) Serial.println("continuing without camera");

  netBegin();

  xTaskCreatePinnedToCore(httpTask,   "http",   8192, NULL, 1, NULL, 0);
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));
}
