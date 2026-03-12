import 'package:flutter/material.dart';

class FooterLink extends StatelessWidget {
  final String label;
  final VoidCallback onTap;

  const FooterLink({
    super.key,
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
