import 'package:flutter/material.dart';

import '../widgets/branding_widgets.dart';
import '../widgets/footer_link.dart';
import 'director_home_page.dart';
import 'player_home_page.dart';

class LandingPage extends StatelessWidget {
  const LandingPage({super.key});

  void _showSignInDialog(BuildContext context) {
    final emailController = TextEditingController();
    final passwordController = TextEditingController();
    String selectedRole = 'Director';

    showDialog<void>(
      context: context,
      builder: (dialogContext) {
        return StatefulBuilder(
          builder: (context, setState) {
            return AlertDialog(
              title: const Text('Sign In'),
              content: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    TextField(
                      controller: emailController,
                      keyboardType: TextInputType.emailAddress,
                      decoration: const InputDecoration(
                        labelText: 'Email Address',
                      ),
                    ),
                    const SizedBox(height: 12),
                    TextField(
                      controller: passwordController,
                      obscureText: true,
                      decoration: const InputDecoration(
                        labelText: 'Password',
                      ),
                    ),
                    const SizedBox(height: 16),
                    RadioListTile<String>(
                      contentPadding: EdgeInsets.zero,
                      title: const Text('Player'),
                      value: 'Player',
                      groupValue: selectedRole,
                      onChanged: (value) {
                        if (value == null) {
                          return;
                        }
                        setState(() => selectedRole = value);
                      },
                    ),
                    RadioListTile<String>(
                      contentPadding: EdgeInsets.zero,
                      title: const Text('Director'),
                      value: 'Director',
                      groupValue: selectedRole,
                      onChanged: (value) {
                        if (value == null) {
                          return;
                        }
                        setState(() => selectedRole = value);
                      },
                    ),
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(dialogContext).pop(),
                  child: const Text('Cancel'),
                ),
                ElevatedButton(
                  onPressed: () {
                    final email = emailController.text.trim();
                    final password = passwordController.text.trim();

                    if (email.isEmpty || password.isEmpty) {
                      ScaffoldMessenger.of(context)
                        ..hideCurrentSnackBar()
                        ..showSnackBar(
                          const SnackBar(
                            content: Text('Please enter both email and password.'),
                          ),
                        );
                      return;
                    }

                    Navigator.of(dialogContext).pop();

                    if (selectedRole == 'Director') {
                      Navigator.of(context).push(
                        MaterialPageRoute(
                          builder: (_) => const SignInHomePage(),
                        ),
                      );
                    } else {
                      Navigator.of(context).push(
                        MaterialPageRoute(
                          builder: (_) => const PlayerSignInHomePage(),
                        ),
                      );
                    }
                  },
                  child: const Text('Continue'),
                ),
              ],
            );
          },
        );
      },
    );
  }

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
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: const Color(0xFF142234),
                    borderRadius: BorderRadius.circular(28),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.4),
                        blurRadius: 24,
                        offset: const Offset(0, 8),
                      ),
                    ],
                  ),
                  child: Image.asset(
                    'assets/worldscore_logo.png',
                    height: 72,
                    fit: BoxFit.contain,
                  ),
                ),
                const SizedBox(height: 36),
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
                Row(
                  children: [
                    Expanded(
                      child: PrimaryButton(
                        label: 'Sign In',
                        backgroundColor: const Color(0xFF1A3A5C),
                        textColor: const Color(0xFF4FC3F7),
                        onPressed: () => _showSignInDialog(context),
                      ),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: PrimaryButton(
                        label: 'Create Account',
                        backgroundColor: const Color(0xFF5A8A1E),
                        textColor: Colors.white,
                        onPressed: () {},
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 48),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    FooterLink(label: 'How It Works', onTap: () {}),
                    FooterLink(label: 'Help & Support', onTap: () {}),
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
