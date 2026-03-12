import 'package:flutter/material.dart';

import 'screens/landing_page.dart';

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
