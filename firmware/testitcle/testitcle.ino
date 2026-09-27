// ============================================================
// TESTITCLE - bench test sketch, Arduino Nano ESP32
// ============================================================
// Reads both VL53L0X sensors and, on boot, parks both haptic servos
// fully inward (600 us) and holds them there.
//
// It can also sweep 600 us <-> 90 deg to exercise the whole travel, but
// that is OFF by default: USB serial and servo power cannot both be
// connected on this rig, so with the servos live there is no way to type
// a command. Set SWEEP_ON_BOOT true, or type 's' when serial is attached.
//
// Geometry for this rig:
//     inward  = low pulse width   -> pressing into the face
//     90 deg  = arm fully OUTWARD -> clear of the face
// Anything past 90 deg is unused, so this sketch refuses to go there.
//
// Units: the LOW end is set in microseconds because that is where the
// servo's mechanical end stop lives and degrees get imprecise down
// there; the HIGH end is set in degrees because that is how the
// linkage was designed. Both are printed together so the mapping is
// visible.
//
// Wiring
//   A4  SDA      -> both VL53L0X
//   A5  SCL      -> both VL53L0X
//   D9  XSHUT    -> LEFT  sensor (becomes I2C 0x30)
//   D8  XSHUT    -> RIGHT sensor (becomes I2C 0x31)
//   D7  SERVO L
//   D6  SERVO R
//
// POWER: do NOT run the servos off the Nano's regulator. Two SG90s
// stalling against an end stop pull well over an amp between them and
// will brown out the board mid-test. Feed them from a separate 5 V
// supply with its ground tied to the Nano's ground.
//
// Serial Monitor at 115200. Type and press Enter:
//     s          start / stop the sweep
//     o          sync <-> opposed (opposed = left presses while right clears)
//     lo <us>    low endpoint in microseconds   (default 600)
//     hi <deg>   high endpoint in degrees       (default 90)
//     t <ms>     half-period in ms              (default 1200)
//     0..90      stop sweeping, send both here
//     l<deg>     stop sweeping, LEFT only
//     r<deg>     stop sweeping, RIGHT only
//     ?          help
// ============================================================

#include <Wire.h>
#include <Adafruit_VL53L0X.h>

// ---------- pins ----------
#define XSHUT_L   D9
#define XSHUT_R   D8
#define ADDR_L    0x30
#define ADDR_R    0x31

#define SERVO_L_PIN D7
#define SERVO_R_PIN D6

// ---------- servo limits ----------
#define ANGLE_MIN   0
#define ANGLE_MAX   90        // never drive past this: the linkage stops here

#define SERVO_MIN_US 500      // reference for the deg <-> us mapping
#define SERVO_MAX_US 2500
#define SERVO_FREQ   50

// 14, NOT 16. SOC_LEDC_TIMER_BIT_WIDE_NUM is 14 on the ESP32-S3, and
// ledcSetup()/ledcAttach() REJECT anything wider: they return 0/false and
// configure no timer at all, so ledcWrite() then writes a duty into an
// unconfigured channel and the pin never toggles. The servos just sit there
// while the sketch's own numbers look perfect.
// This is a RUNTIME limit, so a 16-bit build compiles clean on every core and
// silently does nothing. Trust the boot-time attach report, not the build.
// At 14 bits one LSB is 20000/16384 = 1.22 us, far finer than an SG90's
// ~10 us deadband, so nothing usable is lost.
#define SERVO_BITS   14

// ---------- what it does on boot ----------
// Parks both arms fully inward and holds there. The sweep is OFF by default
// because you cannot have USB serial and servo power connected at the same
// time on this rig, so there is no way to type a command while the servos can
// actually move. Type 's' to sweep when you do have serial attached.
#define SWEEP_ON_BOOT  true

// 600 us, not the 500 us that angleToUs(0) gives. 600 is the inner endpoint
// the sweep has already run to without stalling; 500 is untested and may sit
// against the servo's internal end stop. A stalled SG90 draws ~700 mA and
// damages its own gearbox within minutes, and with serial unplugged you would
// not see it happen. Lower this to 500 only once you have watched it hold at
// 600 quietly, with a hand on the servo body to feel for heat.
#define PARK_US  600

// ---------- sweep defaults ----------
#define SWEEP_LO_US_DEFAULT   600     // innermost, in microseconds
#define SWEEP_HI_DEG_DEFAULT   90     // outermost, in degrees
#define SWEEP_HALF_MS_DEFAULT 1200    // one direction takes this long

#define PRINT_PERIOD_MS 200   // ~5 Hz ToF printout

// ---------- LEDC, both cores ----------
// The Nano ESP32 board exists in BOTH installed platforms, with different LEDC
// APIs, and the Boards menu shows the same name for each:
//   "Arduino ESP32 Boards"  -> arduino:esp32 2.0.x -> ledcSetup + ledcAttachPin
//   "esp32 by Espressif"    -> esp32:esp32  3.0.x -> ledcAttach
// Both cores pin-remap the Dx labels to real GPIOs via io_pin_remap.h, and
// those remaps are macros, so calling them inside these helpers is fine.
#define SERVO_L_CH 0
#define SERVO_R_CH 1

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

// ---------- state ----------
Adafruit_VL53L0X lox_l, lox_r;
bool sensor_l_ok = false, sensor_r_ok = false;

uint32_t usL = PARK_US;
uint32_t usR = PARK_US;

bool     sweeping    = SWEEP_ON_BOOT;
bool     opposed     = false;
uint32_t sweepLoUs   = SWEEP_LO_US_DEFAULT;
uint32_t sweepHiUs   = 0;             // set in setup() from degrees
uint32_t sweepHalfMs = SWEEP_HALF_MS_DEFAULT;
uint32_t sweepStart  = 0;

uint32_t lastPrint = 0;

char rxBuf[32];
uint8_t rxLen = 0;

// ---------- servo math ----------
static inline uint32_t usToDuty(uint32_t us) {
  return (uint32_t)((us * (1UL << SERVO_BITS)) / 20000UL);   // 50 Hz -> 20000 us
}

static inline uint32_t angleToUs(int deg) {
  if (deg < ANGLE_MIN) deg = ANGLE_MIN;
  if (deg > ANGLE_MAX) deg = ANGLE_MAX;
  return SERVO_MIN_US + ((uint32_t)deg * (SERVO_MAX_US - SERVO_MIN_US)) / 180;
}

static inline int usToAngle(uint32_t us) {
  if (us <= SERVO_MIN_US) return 0;
  return (int)(((us - SERVO_MIN_US) * 180UL) / (SERVO_MAX_US - SERVO_MIN_US));
}

// Every servo write funnels through here, so the ANGLE_MAX ceiling cannot be
// bypassed by the sweep, by a typed command, or by a bad endpoint.
void writeUs(int pin, int ch, uint32_t us) {
  uint32_t hi = angleToUs(ANGLE_MAX);
  if (us < SERVO_MIN_US) us = SERVO_MIN_US;
  if (us > hi)           us = hi;
  servoWrite(pin, ch, usToDuty(us));
}

void setLeftUs(uint32_t us)  { usL = us; writeUs(SERVO_L_PIN, SERVO_L_CH, usL); }
void setRightUs(uint32_t us) { usR = us; writeUs(SERVO_R_PIN, SERVO_R_CH, usR); }

// ---------- sweep ----------
void sweepUpdate(uint32_t now) {
  if (!sweeping) return;

  uint32_t period = sweepHalfMs * 2;
  uint32_t phase  = (now - sweepStart) % period;

  // Triangle wave: 0 -> 1 over the first half, 1 -> 0 over the second. Scaled
  // by 1024 instead of floats to keep it cheap and exact.
  uint32_t frac = (phase < sweepHalfMs)
                    ? (phase * 1024UL) / sweepHalfMs
                    : ((period - phase) * 1024UL) / sweepHalfMs;

  uint32_t span = sweepHiUs - sweepLoUs;
  uint32_t a = sweepLoUs + (span * frac) / 1024UL;
  uint32_t b = sweepHiUs - (span * frac) / 1024UL;   // mirror, for opposed mode

  setLeftUs(a);
  setRightUs(opposed ? b : a);
}

void sweepStop(const char *why) {
  if (sweeping) {
    sweeping = false;
    Serial.print("sweep stopped (");
    Serial.print(why);
    Serial.println(")");
  }
}

// ---------- sensors ----------
void sensorsBegin() {
  pinMode(XSHUT_L, OUTPUT);
  pinMode(XSHUT_R, OUTPUT);
  digitalWrite(XSHUT_L, LOW);
  digitalWrite(XSHUT_R, LOW);
  delay(10);

  // Wake LEFT alone and move it off the shared default 0x29, so RIGHT can then
  // claim 0x29 and be renamed without a collision.
  digitalWrite(XSHUT_L, HIGH);
  delay(10);
  sensor_l_ok = lox_l.begin(ADDR_L);

  digitalWrite(XSHUT_R, HIGH);
  delay(10);
  sensor_r_ok = lox_r.begin(ADDR_R);

  Serial.print("ToF L ");
  Serial.print(sensor_l_ok ? "ok" : "FAIL");
  Serial.print("   ToF R ");
  Serial.println(sensor_r_ok ? "ok" : "FAIL");
}

int readRange(Adafruit_VL53L0X &s, bool ok) {
  if (!ok) return -1;
  VL53L0X_RangingMeasurementData_t m;
  s.rangingTest(&m, false);
  return (m.RangeStatus != 4) ? (int)m.RangeMilliMeter : -1;
}

// ---------- serial console ----------
void printUsDeg(uint32_t us) {
  Serial.print(us);
  Serial.print(" us (");
  Serial.print(usToAngle(us));
  Serial.print(" deg)");
}

void printHelp() {
  Serial.println();
  Serial.println("s | o | lo <us> | hi <deg> | t <ms> | 0..90 | l<deg> | r<deg> | ?");
  Serial.print("sweep ");
  Serial.print(sweeping ? "ON" : "off");
  Serial.print(opposed ? " opposed   " : " sync   ");
  printUsDeg(sweepLoUs);
  Serial.print(" <-> ");
  printUsDeg(sweepHiUs);
  Serial.print("   half-period ");
  Serial.print(sweepHalfMs);
  Serial.println(" ms");
}

void reportPositions() {
  Serial.print("-> L ");
  printUsDeg(usL);
  Serial.print("   R ");
  printUsDeg(usR);
  Serial.println();
}

void handleLine(char *line) {
  if (line[0] == '?') { printHelp(); return; }

  if (line[0] == 's' && line[1] == '\0') {
    sweeping = !sweeping;
    sweepStart = millis();
    Serial.print("sweep ");
    Serial.println(sweeping ? "ON" : "off");
    return;
  }
  if (line[0] == 'o' && line[1] == '\0') {
    opposed = !opposed;
    Serial.print("mode: ");
    Serial.println(opposed ? "opposed" : "sync");
    return;
  }
  if (!strncmp(line, "lo", 2)) {
    uint32_t v = (uint32_t)atol(line + 2);
    // Keep it inside the mapping range and strictly below the high end,
    // otherwise the unsigned sweep span underflows.
    if (v < SERVO_MIN_US) v = SERVO_MIN_US;
    if (v >= sweepHiUs)   v = sweepHiUs - 10;
    sweepLoUs = v;
    Serial.print("lo = ");
    printUsDeg(sweepLoUs);
    Serial.println();
    return;
  }
  if (!strncmp(line, "hi", 2)) {
    uint32_t v = angleToUs(atoi(line + 2));
    if (v <= sweepLoUs) v = sweepLoUs + 10;
    sweepHiUs = v;
    Serial.print("hi = ");
    printUsDeg(sweepHiUs);
    Serial.println();
    return;
  }
  if (line[0] == 't') {
    uint32_t v = (uint32_t)atol(line + 1);
    if (v < 100) v = 100;            // below this the servo cannot keep up
    sweepHalfMs = v;
    sweepStart = millis();
    Serial.print("half-period = ");
    Serial.print(sweepHalfMs);
    Serial.println(" ms");
    return;
  }

  if (line[0] == 'l' || line[0] == 'L') {
    sweepStop("manual left");
    setLeftUs(angleToUs(atoi(line + 1)));
  } else if (line[0] == 'r' || line[0] == 'R') {
    sweepStop("manual right");
    setRightUs(angleToUs(atoi(line + 1)));
  } else {
    sweepStop("manual both");
    uint32_t us = angleToUs(atoi(line));
    setLeftUs(us);
    setRightUs(us);
  }
  reportPositions();
}

void pumpSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (rxLen) { rxBuf[rxLen] = '\0'; handleLine(rxBuf); rxLen = 0; }
    } else if (rxLen < sizeof(rxBuf) - 1) {
      rxBuf[rxLen++] = c;
    } else {
      rxLen = 0;
    }
  }
}

// ---------- main ----------
void setup() {
  Serial.begin(115200);
  // No "while (!Serial)" - on USB CDC that blocks forever when the board is
  // not plugged into a computer.

  sweepHiUs = angleToUs(SWEEP_HI_DEG_DEFAULT);

  bool okL = servoAttach(SERVO_L_PIN, SERVO_L_CH);
  bool okR = servoAttach(SERVO_R_PIN, SERVO_R_CH);
  Serial.print("servo PWM  L ");
  Serial.print(okL ? "ok" : "FAILED");
  Serial.print("  R ");
  Serial.print(okR ? "ok" : "FAILED");
  Serial.print("   (");
  Serial.print(SERVO_BITS);
  Serial.print("-bit @ ");
  Serial.print(SERVO_FREQ);
  Serial.println(" Hz)");
  if (!okL || !okR) {
    // A rejected resolution fails at RUNTIME, so the build stays green while
    // the pin never toggles. Check this line before suspecting the hardware.
    Serial.println("!! PWM setup failed - the arms will NOT move. Lower SERVO_BITS.");
  }

  setLeftUs(PARK_US);
  setRightUs(PARK_US);
  Serial.print("parked both arms at ");
  printUsDeg(PARK_US);
  Serial.println();

  Wire.begin();
  Wire.setClock(400000);
  sensorsBegin();

  sweepStart = millis();
  Serial.println("testitcle up");
  printHelp();
}

void loop() {
  pumpSerial();

  uint32_t now = millis();

  // Runs every iteration, not on the ToF cadence: the two blocking range reads
  // cost ~66 ms together, and updating the sweep only that often would make it
  // visibly steppy.
  sweepUpdate(now);

  if (now - lastPrint >= PRINT_PERIOD_MS) {
    lastPrint = now;

    int dl = readRange(lox_l, sensor_l_ok);
    int dr = readRange(lox_r, sensor_r_ok);

    // -1 means out of range or that sensor never initialised. It is not a
    // distance: never treat it as one.
    Serial.print("L ");
    Serial.print(dl);
    Serial.print(" mm  R ");
    Serial.print(dr);
    Serial.print(" mm  |  L ");
    printUsDeg(usL);
    Serial.print("  R ");
    printUsDeg(usR);
    Serial.print("  ");
    Serial.println(sweeping ? (opposed ? "sweep/opp" : "sweep") : "held");
  }
}
