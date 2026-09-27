// ============================================================
// REACHGLASS: glasses controller  (Arduino Nano ESP32 / ESP32-S3)
// ============================================================
// Owns: 2x VL53L0X ToF sensors, 2 haptic press servos, and its OWN
// WiFi link. There is NO UART to the ESP32-CAM any more - the Nano
// joins the CAM's access point as a station and talks UDP straight to
// the laptop. The CAM is now only a camera + access point.
//
// Why this is better than the old UART bridge: no wiring between the
// two boards, no framing protocol, and servo commands no longer queue
// behind the bridge's task. The two data paths are fully independent.
//
// Wiring
//   A4  SDA      -> both VL53L0X SDA
//   A5  SCL      -> both VL53L0X SCL
//   D9  XSHUT    -> LEFT  sensor (becomes I2C 0x30)
//   D8  XSHUT    -> RIGHT sensor (becomes I2C 0x31)
//   D6  SERVO R  -> haptic pad pressing the RIGHT side of the face
//   D7  SERVO L  -> haptic pad pressing the LEFT side of the face
//   (no connection to the ESP32-CAM at all)
//
// POWER: do NOT run the servos off the Nano's regulator. Two SG90s
// stalling pull well over an amp between them and will brown out the
// board. Separate 5 V supply, grounds tied.
//
// Network
//   The CAM runs the AP. This board joins it as a station on a static
//   address, so the laptop always knows where to find it:
//     CAM  (video)  192.168.4.1   http://192.168.4.1/stream
//     Nano (data)   192.168.4.50  UDP 4210
//   The laptop must send at least one packet first; that is how this
//   board learns where to send telemetry. It also broadcasts a
//   discovery beacon on UDP 4211 until a peer registers.
//
// Protocol v1 (ASCII, newline terminated, one message per datagram)
//   out, ~15 Hz:
//     T,<seq>,<uptime_ms>,<tof_l_mm>,<tof_r_mm>,<press_l>,<press_r>,<link>
//       seq wraps; tof -1 = out of range or sensor dead; press 0..100;
//       link 1 = a command arrived within LINK_TIMEOUT_MS.
//   in:
//     D,<dir>[,<depth>]                      direction cue (PRIMARY, see below)
//     H,<left>,<right>                       hold press depth 0..100 each
//     P,<side>,<count>,<on>,<off>,<depth>     pulse burst, side = L|R|B
//     Z                                      release both now (safety stop)
//     C,<side>,<released_deg>,<pressed_deg>   runtime trim, not persisted
//     S,<left_deg>,<right_deg>               raw angles, bench use only
//
// THE DIRECTION CUE, and its deliberately crossed sides:
//     D,-1  = go LEFT   -> presses the RIGHT pad
//     D,+1  = go RIGHT  -> presses the LEFT pad
//     D, 0  = release both
// The pad on the side OPPOSITE the turn is the one that presses, so the wearer
// feels a nudge from the far side pushing them the way they should go. This is
// the inverse of the earlier convention, which pressed the near side. Only one
// pad is ever driven by a D command - pressing both is not a direction.
//
// Failsafe: LINK_TIMEOUT_MS of command silence releases both pads to
// 0%. It deliberately does NOT hold the last position - a servo latched
// against a wearer's face after a WiFi drop is the worst failure here.
//
// Library needed: "Adafruit_VL53L0X" (pulls in Adafruit BusIO).
// Servos use the ESP32 LEDC peripheral directly - no servo library.
// ============================================================

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <Adafruit_VL53L0X.h>

// ---------- network ----------
const char *AP_SSID = "rover";          // must match cam_bridge.ino
const char *AP_PASS = "rover1234";

IPAddress MY_IP  (192, 168, 4, 50);     // static: clear of the AP's DHCP pool
IPAddress MY_GW  (192, 168, 4, 1);      // the CAM
IPAddress MY_MASK(255, 255, 255, 0);

#define DATA_PORT       4210
#define DISCOVERY_PORT  4211
#define BEACON_PERIOD_MS 1000
#define REJOIN_PERIOD_MS 3000

// ---------- sensors ----------
#define XSHUT_L   D9
#define XSHUT_R   D8
#define ADDR_L    0x30
#define ADDR_R    0x31

// ---------- servos ----------
#define SERVO_L_PIN D7
#define SERVO_R_PIN D6

// Measured on the rig: the arm swings INWARD toward the face as the pulse
// gets shorter. 90 deg is fully clear, 0 deg is hardest press - so pressed
// sits BELOW released and the span is negative. applyPress() handles that
// sign; do not "fix" it by swapping them.
#define L_RELEASED_DEG  90
#define L_PRESSED_DEG    0
#define R_RELEASED_DEG  90
#define R_PRESSED_DEG    0

// Hard floor on pulse width, enforced on EVERY servo write. 0 deg maps to
// 500 us, but 500 may sit against the servo's internal end stop, where an
// SG90 stalls, draws ~700 mA and cooks its own gearbox. 600 us is the value
// bench-tested on this mechanism (see testitcle). Lower it only after
// watching the arm hold at 600 quietly and cool.
#define SERVO_FLOOR_US 600

// Compile-time hard stop on travel, in degrees away from the released angle.
// 90 permits the whole mechanical range and therefore gives no margin.
// LOWER IT once someone has worn the rig and found what is comfortable:
// this constant is what stops a runaway command hurting the wearer.
#define MAX_PRESS_TRAVEL_DEG  90

#define SERVO_MIN_US 500        // reference for the deg <-> us mapping
#define SERVO_MAX_US 2500
#define SERVO_FREQ   50

// 14, NOT 16. SOC_LEDC_TIMER_BIT_WIDE_NUM is 14 on the ESP32-S3, and
// ledcSetup()/ledcAttach() REJECT anything wider: they return 0/false and
// configure no timer at all, so ledcWrite() writes a duty into an
// unconfigured channel and the pin never toggles. The pads sit still while
// every number in the firmware looks perfect. This is a RUNTIME limit, so a
// 16-bit build compiles clean on both cores and silently does nothing -
// which is exactly how it was missed. Trust the boot-time attach report.
// At 14 bits one LSB is 20000/16384 = 1.22 us, far finer than an SG90's
// ~10 us deadband.
#define SERVO_BITS   14

// The Nano ESP32 board exists in BOTH installed ESP32 platforms, with
// different LEDC APIs, and the Boards menu shows the same name for each:
//   "Arduino ESP32 Boards"  -> arduino:esp32 2.0.x -> ledcSetup + ledcAttachPin
//   "esp32 by Espressif"    -> esp32:esp32  3.0.x -> ledcAttach
// Both cores pin-remap the Dx labels to real GPIOs via io_pin_remap.h, and
// those remaps are macros, so calling them inside these helpers is fine.
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
static inline bool servoAttach(int pin, int ch) {
  (void)ch;                                  // 3.x allocates the channel itself
  return ledcAttach(pin, SERVO_FREQ, SERVO_BITS);
}
static inline void servoWrite(int pin, int ch, uint32_t duty) {
  (void)ch;
  ledcWrite(pin, duty);
}
#else
static inline bool servoAttach(int pin, int ch) {
  if (ledcSetup(ch, SERVO_FREQ, SERVO_BITS) == 0) return false;   // 0 = rejected
  ledcAttachPin(pin, ch);
  return true;
}
static inline void servoWrite(int pin, int ch, uint32_t duty) {
  (void)pin;
  ledcWrite(ch, duty);                       // 2.x writes by CHANNEL, not pin
}
#endif

// ---------- timing ----------
#define LINK_TIMEOUT_MS 600   // no command for this long -> release both
#define TELEM_PERIOD_MS 50    // asks for 20 Hz; the blocking ToF reads
                              // floor the real rate at ~15 Hz
#define PULSE_MIN_MS    10
#define PULSE_MAX_COUNT 255

// Press depth a bare "D,<dir>" uses. The upstream signal is discrete (-1/0/+1)
// with no magnitude, so there is nothing to scale and one fixed depth is the
// whole cue. 100 means "as hard as MAX_PRESS_TRAVEL_DEG allows" - turn it down
// here once someone has worn the rig, or override per command with D,<dir>,<depth>.
#define DIR_PRESS_PCT 100

// ---------- state ----------
enum { SIDE_L = 0, SIDE_R = 1, NUM_SIDES = 2 };
#define MASK_L 0x1
#define MASK_R 0x2

Adafruit_VL53L0X lox_l, lox_r;
bool sensor_l_ok = false, sensor_r_ok = false;
bool pwmOk = false;

const int SERVO_PIN[NUM_SIDES] = { SERVO_L_PIN, SERVO_R_PIN };
const int SERVO_CH [NUM_SIDES] = { 0, 1 };   // only used on core 2.x

int  releasedDeg[NUM_SIDES] = { L_RELEASED_DEG, R_RELEASED_DEG };
int  pressedDeg [NUM_SIDES] = { L_PRESSED_DEG,  R_PRESSED_DEG  };
int  holdPct [NUM_SIDES] = { 0, 0 };   // setpoint from the last H command
int  pressPct[NUM_SIDES] = { 0, 0 };   // what is actually applied right now

WiFiUDP udp, beacon;
bool      havePeer = false;
IPAddress peerIP;
uint16_t  peerPort = 0;

uint32_t seq         = 0;
uint32_t lastCmdMs   = 0;
uint32_t lastTelemMs = 0;
uint32_t lastBeacon  = 0;
uint32_t lastRejoin  = 0;
bool     failsafed   = false;

struct Burst {
  bool     active     = false;
  uint8_t  mask       = 0;
  uint16_t remaining  = 0;
  uint32_t onMs       = 0;
  uint32_t offMs      = 0;
  int      depth      = 0;
  bool     phaseOn    = false;
  uint32_t phaseStart = 0;
} burst;

char rxBuf[128];

// ---------- servos ----------
static inline uint32_t usToDuty(uint32_t us) {
  return (uint32_t)((us * (1UL << SERVO_BITS)) / 20000UL);   // 50 Hz period
}

void servoWriteAngle(int side, int angle) {
  if (angle < 0)   angle = 0;
  if (angle > 180) angle = 180;
  uint32_t us = SERVO_MIN_US + ((uint32_t)angle * (SERVO_MAX_US - SERVO_MIN_US)) / 180;
  if (us < SERVO_FLOOR_US) us = SERVO_FLOOR_US;   // never stall on the end stop
  servoWrite(SERVO_PIN[side], SERVO_CH[side], usToDuty(us));
}

// The single place press depth turns into an angle, so the travel clamp
// cannot be bypassed by any command path except raw S.
void applyPress(int side, int pct) {
  if (pct < 0)   pct = 0;
  if (pct > 100) pct = 100;

  int span   = pressedDeg[side] - releasedDeg[side];
  int travel = (span * pct) / 100;
  if (travel >  MAX_PRESS_TRAVEL_DEG) travel =  MAX_PRESS_TRAVEL_DEG;
  if (travel < -MAX_PRESS_TRAVEL_DEG) travel = -MAX_PRESS_TRAVEL_DEG;

  pressPct[side] = pct;
  servoWriteAngle(side, releasedDeg[side] + travel);
}

void applyMaskPress(uint8_t mask, int pct) {
  if (mask & MASK_L) applyPress(SIDE_L, pct);
  if (mask & MASK_R) applyPress(SIDE_R, pct);
}

void applyMaskHold(uint8_t mask) {
  if (mask & MASK_L) applyPress(SIDE_L, holdPct[SIDE_L]);
  if (mask & MASK_R) applyPress(SIDE_R, holdPct[SIDE_R]);
}

void burstCancel() { burst.active = false; }

void releaseAll() {
  burstCancel();
  for (int s = 0; s < NUM_SIDES; s++) { holdPct[s] = 0; applyPress(s, 0); }
}

void servosBegin() {
  pwmOk = true;
  for (int s = 0; s < NUM_SIDES; s++) {
    if (!servoAttach(SERVO_PIN[s], SERVO_CH[s])) pwmOk = false;
    applyPress(s, 0);
  }
  // Report it: a rejected resolution fails at RUNTIME, so the build stays
  // green while the pads never move. Check this before the mechanism.
  Serial.printf("servo PWM %s (%d-bit @ %d Hz, floor %d us)\n",
                pwmOk ? "ok" : "FAILED - pads will NOT move, lower SERVO_BITS",
                SERVO_BITS, SERVO_FREQ, SERVO_FLOOR_US);
}

// ---------- pulse burst ----------
void burstStart(uint8_t mask, int count, uint32_t onMs, uint32_t offMs, int depth) {
  burstCancel();
  if (!mask || count <= 0) { applyMaskHold(mask); return; }
  if (count > PULSE_MAX_COUNT) count = PULSE_MAX_COUNT;
  if (onMs  < PULSE_MIN_MS) onMs  = PULSE_MIN_MS;
  if (offMs < PULSE_MIN_MS) offMs = PULSE_MIN_MS;

  burst.active     = true;
  burst.mask       = mask;
  burst.remaining  = (uint16_t)(count - 1);
  burst.onMs       = onMs;
  burst.offMs      = offMs;
  burst.depth      = depth;
  burst.phaseOn    = true;
  burst.phaseStart = millis();
  applyMaskPress(mask, depth);
}

// Never blocks, so telemetry and command handling keep full rate.
void burstUpdate(uint32_t now) {
  if (!burst.active) return;
  uint32_t el = now - burst.phaseStart;

  if (burst.phaseOn) {
    if (el < burst.onMs) return;
    applyMaskHold(burst.mask);              // the gap sits at the held depth
    if (burst.remaining == 0) { burst.active = false; return; }
    burst.phaseOn    = false;
    burst.phaseStart = now;
  } else {
    if (el < burst.offMs) return;
    burst.remaining--;
    burst.phaseOn    = true;
    burst.phaseStart = now;
    applyMaskPress(burst.mask, burst.depth);
  }
}

// ---------- sensors ----------
void sensorsBegin() {
  pinMode(XSHUT_L, OUTPUT);
  pinMode(XSHUT_R, OUTPUT);
  digitalWrite(XSHUT_L, LOW);
  digitalWrite(XSHUT_R, LOW);
  delay(10);

  // Wake LEFT alone and move it off the shared default 0x29, so RIGHT can
  // then claim 0x29 and be renamed without a collision.
  digitalWrite(XSHUT_L, HIGH);
  delay(10);
  sensor_l_ok = lox_l.begin(ADDR_L);

  digitalWrite(XSHUT_R, HIGH);
  delay(10);
  sensor_r_ok = lox_r.begin(ADDR_R);

  Serial.printf("ToF L %s  ToF R %s\n", sensor_l_ok ? "ok" : "FAIL",
                                        sensor_r_ok ? "ok" : "FAIL");
}

int readRange(Adafruit_VL53L0X &s, bool ok) {
  if (!ok) return -1;
  VL53L0X_RangingMeasurementData_t m;
  s.rangingTest(&m, false);
  return (m.RangeStatus != 4) ? (int)m.RangeMilliMeter : -1;
}

// ---------- command parsing ----------
static uint8_t sideMask(const char *tok) {
  if (!tok) return 0;
  switch (tok[0]) {
    case 'L': case 'l': return MASK_L;
    case 'R': case 'r': return MASK_R;
    case 'B': case 'b': return MASK_L | MASK_R;
    default:            return 0;
  }
}

// Continues strtok over the line already handed to handleLine().
//
// CAUTION: this ADVANCES strtok, so it must never appear inside constrain()
// or any other macro. constrain() is
//     ((a) < (lo) ? (lo) : ((a) > (hi) ? (hi) : (a)))
// which evaluates its first argument up to THREE times - so
// constrain(nextInt(0),..) silently eats the next two fields and returns the
// default. That bug made P ignore its depth and C zero both angles. Always
// read into a local first, then clamp the local.
static int nextInt(int def) {
  char *t = strtok(NULL, ",");
  return t ? atoi(t) : def;
}

void handleLine(char *line) {
  char kind = line[0];
  strtok(line, ",");               // consume the command letter itself
  bool known = true;

  switch (kind) {
    case 'D': {                    // D,<dir>[,<depth>]   dir = -1 | 0 | +1
      burstCancel();
      int d     = nextInt(0);
      int depth = nextInt(DIR_PRESS_PCT);
      depth = constrain(depth, 0, 100);
      // Crossed on purpose: +1 (go right) presses LEFT, -1 (go left) presses
      // RIGHT. Two independent tests, so a malformed dir can never light both
      // pads - the one thing that would be unreadable as a direction.
      //
      // Only the SIGN is used, not the magnitude: D,7 presses left just like
      // D,1. That is deliberate. If something upstream ever sends a raw bearing
      // here instead of -1/0/+1, a correct-direction cue is a much softer
      // failure than releasing and leaving the wearer with no guidance. The
      // host clamps to -1/0/+1 anyway.
      holdPct[SIDE_L] = (d > 0) ? depth : 0;
      holdPct[SIDE_R] = (d < 0) ? depth : 0;
      applyMaskHold(MASK_L | MASK_R);
      break;
    }

    case 'H': {                    // H,<left>,<right>
      burstCancel();
      int l = nextInt(0), r = nextInt(0);
      holdPct[SIDE_L] = constrain(l, 0, 100);
      holdPct[SIDE_R] = constrain(r, 0, 100);
      applyMaskHold(MASK_L | MASK_R);
      break;
    }

    case 'P': {                    // P,<side>,<count>,<on>,<off>,<depth>
      uint8_t m = sideMask(strtok(NULL, ","));
      int count = nextInt(0);
      int onMs  = nextInt(80);
      int offMs = nextInt(80);
      int depth = nextInt(100);
      depth = constrain(depth, 0, 100);
      burstStart(m, count, (uint32_t)max(0, onMs), (uint32_t)max(0, offMs), depth);
      break;
    }

    case 'Z':                      // safety stop
      releaseAll();
      break;

    case 'C': {                    // C,<side>,<released_deg>,<pressed_deg>
      uint8_t m = sideMask(strtok(NULL, ","));
      int rel = nextInt(0);
      int prs = nextInt(0);
      rel = constrain(rel, 0, 180);
      prs = constrain(prs, 0, 180);
      for (int s = 0; s < NUM_SIDES; s++) {
        if (m & (s == SIDE_L ? MASK_L : MASK_R)) {
          releasedDeg[s] = rel;
          pressedDeg[s]  = prs;
          applyPress(s, pressPct[s]);   // re-seat at the new geometry
        }
      }
      break;
    }

    case 'S': {                    // raw angles, bypasses the press model
      burstCancel();
      int la = nextInt(-1), ra = nextInt(-1);
      // pressPct no longer describes the pad, so report 0 rather than lie.
      if (la >= 0) { servoWriteAngle(SIDE_L, la); pressPct[SIDE_L] = 0; }
      if (ra >= 0) { servoWriteAngle(SIDE_R, ra); pressPct[SIDE_R] = 0; }
      break;
    }

    default:
      known = false;               // garbage must not feed the watchdog
      break;
  }

  if (known) { lastCmdMs = millis(); failsafed = false; }
}

// ---------- network ----------
void netBegin() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);            // power save adds 100 ms+ of RX jitter
  WiFi.config(MY_IP, MY_GW, MY_MASK);
  WiFi.begin(AP_SSID, AP_PASS);

  Serial.printf("joining \"%s\"", AP_SSID);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 12000) {
    delay(300);
    Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("joined, data on %s:%d\n",
                  WiFi.localIP().toString().c_str(), DATA_PORT);
  } else {
    // Do NOT spin here. The pads must stay releasable and the sketch must
    // keep running with the AP down; loop() retries in the background.
    Serial.println("join failed - retrying in the background, pads stay released");
  }
  udp.begin(DATA_PORT);
}

// Announce where we are until the laptop starts talking to us.
void sendBeacon() {
  char msg[64];
  int n = snprintf(msg, sizeof(msg), "ROVER,%s,%u",
                   WiFi.localIP().toString().c_str(), (unsigned)DATA_PORT);
  beacon.beginPacket(IPAddress(255, 255, 255, 255), DISCOVERY_PORT);
  beacon.write((const uint8_t *)msg, n);
  beacon.endPacket();
}

void pumpNet() {
  int n = udp.parsePacket();
  while (n > 0) {
    peerIP   = udp.remoteIP();
    peerPort = udp.remotePort();
    havePeer = true;

    int got = udp.read(rxBuf, sizeof(rxBuf) - 1);
    if (got > 0) {
      rxBuf[got] = '\0';
      // One command per datagram, but tolerate a trailing newline and a
      // sender that batches a few lines together.
      char *save = NULL;
      for (char *ln = strtok_r(rxBuf, "\r\n", &save); ln; ln = strtok_r(NULL, "\r\n", &save)) {
        if (*ln) handleLine(ln);
      }
    }
    n = udp.parsePacket();
  }
}

// ---------- main ----------
void setup() {
  Serial.begin(115200);          // USB debug only. Do NOT wait on it:
                                 // while(!Serial) hangs forever untethered.
  servosBegin();                 // pads released before anything else runs

  Wire.begin();
  Wire.setClock(400000);
  sensorsBegin();

  netBegin();

  lastCmdMs = millis();
  Serial.println("reachglass controller up");
}

void loop() {
  pumpNet();

  uint32_t now = millis();

  // Rejoin in the background. WiFi.begin() is not re-issued on every pass:
  // hammering it prevents the supplicant from ever completing.
  if (WiFi.status() != WL_CONNECTED && now - lastRejoin >= REJOIN_PERIOD_MS) {
    lastRejoin = now;
    havePeer = false;
    WiFi.begin(AP_SSID, AP_PASS);
  }

  bool link = (now - lastCmdMs) < LINK_TIMEOUT_MS;
  if (!link && !failsafed) { releaseAll(); failsafed = true; }

  burstUpdate(now);

  if (!havePeer && now - lastBeacon >= BEACON_PERIOD_MS) {
    lastBeacon = now;
    if (WiFi.status() == WL_CONNECTED) sendBeacon();
  }

  if (now - lastTelemMs >= TELEM_PERIOD_MS) {
    lastTelemMs = now;

    int dl = readRange(lox_l, sensor_l_ok);
    pumpNet();                       // each ranging call costs ~33 ms;
    int dr = readRange(lox_r, sensor_r_ok);
    pumpNet();                       // stay responsive across both

    char line[96];
    int n = snprintf(line, sizeof(line), "T,%lu,%lu,%d,%d,%d,%d,%d\n",
                     (unsigned long)seq++, (unsigned long)millis(),
                     dl, dr, pressPct[SIDE_L], pressPct[SIDE_R], link ? 1 : 0);

    if (havePeer && WiFi.status() == WL_CONNECTED) {
      udp.beginPacket(peerIP, peerPort);
      udp.write((const uint8_t *)line, n);
      udp.endPacket();
    } else {
      Serial.print(line);           // no peer yet: still observable over USB
    }
  }
}
