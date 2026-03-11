import 'package:flutter/material.dart';

void main() {
  runApp(const WorldScoreAIApp());
}

class WorldScoreAIApp extends StatelessWidget {
  const WorldScoreAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'WorldScoreAI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0A1628)),
        useMaterial3: true,
      ),
      home: const LandingPage(),
    );
  }
}

class LandingPage extends StatelessWidget {
  const LandingPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1B2A),
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 40.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                const SizedBox(height: 20),

                // Logo Card
                _LogoCard(),

                const SizedBox(height: 36),

                // Tagline
                const Text(
                  'Snap. Score. Track.',
                  style: TextStyle(
                    color: Color(0xFF4FC3F7),
                    fontSize: 26,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 0.5,
                  ),
                  textAlign: TextAlign.center,
                ),

                const SizedBox(height: 14),

                // Subtitle
                const Text(
                  'Snap, score, and track golf rounds instantly — the mobile app trusted by golfers, clubs, and pro shops.',
                  style: TextStyle(
                    color: Color(0xFFB0BEC5),
                    fontSize: 14.5,
                    fontWeight: FontWeight.w400,
                    height: 1.6,
                  ),
                  textAlign: TextAlign.center,
                ),

                const SizedBox(height: 36),

                // Sign In / Create Account Buttons
                Row(
                  children: [
                    Expanded(
                      child: _PrimaryButton(
                        label: 'Sign In',
                        backgroundColor: const Color(0xFF1A3A5C),
                        textColor: const Color(0xFF4FC3F7),
                        onPressed: () {},
                      ),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: _PrimaryButton(
                        label: 'Create Account',
                        backgroundColor: const Color(0xFF5A8A1E),
                        textColor: Colors.white,
                        onPressed: () {},
                      ),
                    ),
                  ],
                ),


                const SizedBox(height: 48),

                // Footer Links
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    _FooterLink(label: 'View Plans', onTap: () {}),
                    _FooterLink(label: 'How It Works', onTap: () {}),
                    _FooterLink(label: 'Help & Support', onTap: () {}),
                  ],
                ),

                const SizedBox(height: 20),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ──────────────────────────────────────────
// Logo Card Widget
// ──────────────────────────────────────────
class _LogoCard extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      width: 160,
      height: 160,
      decoration: BoxDecoration(
        color: const Color(0xFF142234),
        borderRadius: BorderRadius.circular(28),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.4),
            blurRadius: 24,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Globe icon with checkmark
          SizedBox(
            width: 72,
            height: 72,
            child: CustomPaint(
              painter: _GlobeCheckPainter(),
            ),
          ),
          const SizedBox(height: 12),
          const Text(
            'WorldScoreAI',
            style: TextStyle(
              color: Colors.white,
              fontSize: 13,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.3,
            ),
          ),
        ],
      ),
    );
  }
}

// ──────────────────────────────────────────
// Globe + Checkmark Custom Painter
// ──────────────────────────────────────────
class _GlobeCheckPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final double cx = size.width / 2;
    final double cy = size.height / 2;
    final double r = size.width * 0.44;

    final linePaint = Paint()
      ..color = Colors.white.withOpacity(0.85)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.8
      ..strokeCap = StrokeCap.round;

    // Outer circle (globe outline)
    canvas.drawCircle(Offset(cx, cy), r, linePaint);

    // Vertical center line (meridian)
    canvas.drawLine(Offset(cx, cy - r), Offset(cx, cy + r), linePaint);

    // Horizontal lines (latitude bands)
    for (final double fraction in [-0.45, 0.0, 0.45]) {
      final double y = cy + fraction * r * 1.3;
      final double halfW = _chordHalfWidth(r, y - cy);
      canvas.drawLine(
        Offset(cx - halfW, y),
        Offset(cx + halfW, y),
        linePaint,
      );
    }

    // Curved vertical lines (longitudes left & right)
    _drawLongitudeCurve(canvas, cx, cy, r, -0.55, linePaint);
    _drawLongitudeCurve(canvas, cx, cy, r, 0.55, linePaint);

    // Green checkmark overlay (top-right area)
    final checkBgPaint = Paint()
      ..color = const Color(0xFF5A8A1E)
      ..style = PaintingStyle.fill;

    final double checkCx = cx + r * 0.35;
    final double checkCy = cy - r * 0.35;
    const double checkR = 10.0;

    canvas.drawCircle(Offset(checkCx, checkCy), checkR, checkBgPaint);

    final checkPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final path = Path()
      ..moveTo(checkCx - 5, checkCy)
      ..lineTo(checkCx - 1.5, checkCy + 4)
      ..lineTo(checkCx + 5, checkCy - 4);

    canvas.drawPath(path, checkPaint);
  }

  double _chordHalfWidth(double r, double dy) {
    final double val = r * r - dy * dy;
    return val <= 0 ? 0 : (val < 0 ? 0 : val == 0 ? 0 : (val).abs().clamp(0, double.infinity) > 0 ? (val < 0 ? 0 : _sqrt(val)) : 0);
  }

  double _sqrt(double val) => val <= 0 ? 0 : (val * val > 0 ? val.abs() : 0) == 0 ? 0 : _mySqrt(val);

  double _mySqrt(double x) {
    // Simple Newton's method sqrt
    if (x <= 0) return 0;
    double z = x / 2;
    for (int i = 0; i < 20; i++) {
      z = (z + x / z) / 2;
    }
    return z;
  }

  void _drawLongitudeCurve(
      Canvas canvas, double cx, double cy, double r, double xOffset, Paint paint) {
    final path = Path();
    final double ctrlX = cx + xOffset * r * 2.0;
    path.moveTo(cx, cy - r);
    path.quadraticBezierTo(ctrlX, cy, cx, cy + r);
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

// ──────────────────────────────────────────
// Primary Button Widget
// ──────────────────────────────────────────
class _PrimaryButton extends StatelessWidget {
  final String label;
  final Color backgroundColor;
  final Color textColor;
  final VoidCallback onPressed;

  const _PrimaryButton({
    required this.label,
    required this.backgroundColor,
    required this.textColor,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 50,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: backgroundColor,
          foregroundColor: textColor,
          elevation: 2,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: textColor,
            fontSize: 15,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.3,
          ),
        ),
      ),
    );
  }
}

// ──────────────────────────────────────────
// Footer Link Widget
// ──────────────────────────────────────────
class _FooterLink extends StatelessWidget {
  final String label;
  final VoidCallback onTap;

  const _FooterLink({
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Text(
        label,
        style: const TextStyle(
          color: Color(0xFF607D8B),
          fontSize: 12.5,
          fontWeight: FontWeight.w400,
          decoration: TextDecoration.none,
        ),
      ),
    );
  }
}