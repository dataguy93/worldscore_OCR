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
                        onPressed: () => _showSignInDialog(context),
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

class PlayerSignInHomePage extends StatelessWidget {
  const PlayerSignInHomePage({super.key});

  static const double _headerBarHeight = 64;

  void _showMenuSelection(BuildContext context, String value) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          content: Text('$value selected'),
          duration: const Duration(seconds: 2),
        ),
      );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1B2A),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 22.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Row(
                        children: [
                          Expanded(
                            child: Container(
                              height: _headerBarHeight,
                              padding: const EdgeInsets.symmetric(horizontal: 18),
                              alignment: Alignment.centerLeft,
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(16),
                                gradient: const LinearGradient(
                                  colors: [Color(0xFF1A2E44), Color(0xFF223F5E)],
                                  begin: Alignment.topLeft,
                                  end: Alignment.bottomRight,
                                ),
                                border: Border.all(color: const Color(0xFF355C84)),
                              ),
                              child: const Text(
                                'WORLDSCORE AI',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 24,
                                  fontWeight: FontWeight.w700,
                                  letterSpacing: 1.0,
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(width: 10),
                          PopupMenuButton<String>(
                            tooltip: 'Open menu',
                            onSelected: (value) => _showMenuSelection(context, value),
                            color: const Color(0xFF142234),
                            position: PopupMenuPosition.under,
                            offset: const Offset(0, 8),
                            itemBuilder: (context) => const [
                              PopupMenuItem(
                                value: 'Account',
                                child: Text('Account', style: TextStyle(color: Colors.white)),
                              ),
                              PopupMenuItem(
                                value: 'Who We Are',
                                child: Text('Who We Are', style: TextStyle(color: Colors.white)),
                              ),
                              PopupMenuItem(
                                value: 'FAQ',
                                child: Text('FAQ', style: TextStyle(color: Colors.white)),
                              ),
                              PopupMenuItem(
                                value: 'Settings',
                                child: Text('Settings', style: TextStyle(color: Colors.white)),
                              ),
                            ],
                            child: Container(
                              height: _headerBarHeight,
                              alignment: Alignment.center,
                              decoration: BoxDecoration(
                                color: const Color(0xFF294B6D),
                                borderRadius: BorderRadius.circular(10),
                              ),
                              padding: const EdgeInsets.symmetric(horizontal: 14),
                              child: const Icon(
                                Icons.menu,
                                color: Colors.white,
                                size: 22,
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 28),
                      const Text(
                        'Welcome back',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: Color(0xFFB8C7D6),
                          fontSize: 15,
                          fontWeight: FontWeight.w500,
                          letterSpacing: 0.2,
                        ),
                      ),
                      const SizedBox(height: 20),
                      const _PlayerOverviewCard(),
                      const SizedBox(height: 20),
                      const _MenuCard(
                        label: 'Leaderboard',
                        subtitle: 'See current and former tournament standings.',
                      ),
                      const SizedBox(height: 14),
                      const _MenuCard(
                        label: 'Round History',
                        subtitle: 'Review your round history and submitted scorecards.',
                      ),
                      const SizedBox(height: 14),
                      const _MenuCard(
                        label: 'Upload',
                        subtitle: 'Submit a new scorecard using AI OCR.',
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 28),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  _FooterLink(label: 'How It Works', onTap: () {}),
                  _FooterLink(label: 'Help & Support', onTap: () {}),
                ],
              ),
              const SizedBox(height: 8),
            ],
          ),
        ),
      ),
    );
  }
}

class _PlayerOverviewCard extends StatelessWidget {
  const _PlayerOverviewCard();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF142234),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFF1F3A56)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Player Snapshot',
            style: TextStyle(
              color: Color(0xFF4FC3F7),
              fontSize: 17,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(
                child: Container(
                  height: 112,
                  decoration: BoxDecoration(
                    color: const Color(0xFF0F1A2A),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: const Color(0xFF2A4D70)),
                  ),
                  child: const Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.add_a_photo_outlined, color: Color(0xFF7FA6C9), size: 28),
                      SizedBox(height: 8),
                      Text(
                        'Upload photo',
                        style: TextStyle(
                          color: Color(0xFF9FB3C8),
                          fontSize: 13,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 12),
              const Expanded(
                flex: 2,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _PlayerInfoRow(label: 'Name', value: 'Alex Morgan'),
                    SizedBox(height: 8),
                    _PlayerInfoRow(label: 'Rounds this year', value: '24'),
                    SizedBox(height: 8),
                    _PlayerInfoRow(label: 'Average score', value: '83.4'),
                    SizedBox(height: 8),
                    _PlayerInfoRow(label: 'Handicap', value: '12.6'),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _PlayerInfoRow extends StatelessWidget {
  final String label;
  final String value;

  const _PlayerInfoRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return RichText(
      text: TextSpan(
        text: '$label: ',
        style: const TextStyle(
          color: Color(0xFF9FB3C8),
          fontSize: 13,
          fontWeight: FontWeight.w500,
        ),
        children: [
          TextSpan(
            text: value,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

class SignInHomePage extends StatelessWidget {
  const SignInHomePage({super.key});

  static const double _headerBarHeight = 64;

  void _showMenuSelection(BuildContext context, String value) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          content: Text('$value selected'),
          duration: const Duration(seconds: 2),
        ),
      );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1B2A),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 22.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Row(
                        children: [
                          Expanded(
                            child: Container(
                              height: _headerBarHeight,
                              padding: const EdgeInsets.symmetric(horizontal: 18),
                              alignment: Alignment.centerLeft,
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(16),
                                gradient: const LinearGradient(
                                  colors: [Color(0xFF1A2E44), Color(0xFF223F5E)],
                                  begin: Alignment.topLeft,
                                  end: Alignment.bottomRight,
                                ),
                                border: Border.all(color: const Color(0xFF355C84)),
                              ),
                              child: const Text(
                                'WORLDSCORE AI',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 24,
                                  fontWeight: FontWeight.w700,
                                  letterSpacing: 1.0,
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(width: 10),
                          PopupMenuButton<String>(
                            tooltip: 'Open menu',
                            onSelected: (value) => _showMenuSelection(context, value),
                            color: const Color(0xFF142234),
                            position: PopupMenuPosition.under,
                            offset: const Offset(0, 8),
                            itemBuilder: (context) => const [
                              PopupMenuItem(
                                value: 'Account',
                                child: Text('Account', style: TextStyle(color: Colors.white)),
                              ),
                              PopupMenuItem(
                                value: 'Who We Are',
                                child: Text('Who We Are', style: TextStyle(color: Colors.white)),
                              ),
                              PopupMenuItem(
                                value: 'FAQ',
                                child: Text('FAQ', style: TextStyle(color: Colors.white)),
                              ),
                              PopupMenuItem(
                                value: 'Settings',
                                child: Text('Settings', style: TextStyle(color: Colors.white)),
                              ),
                            ],
                            child: Container(
                              height: _headerBarHeight,
                              alignment: Alignment.center,
                              decoration: BoxDecoration(
                                color: const Color(0xFF294B6D),
                                borderRadius: BorderRadius.circular(10),
                              ),
                              padding: const EdgeInsets.symmetric(horizontal: 14),
                              child: const Icon(
                                Icons.menu,
                                color: Colors.white,
                                size: 22,
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 28),
                      const Text(
                        'Welcome back, Director',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: Color(0xFFB8C7D6),
                          fontSize: 15,
                          fontWeight: FontWeight.w500,
                          letterSpacing: 0.2,
                        ),
                      ),
                      const SizedBox(height: 20),
                      const _DirectorOverviewCard(),
                      const SizedBox(height: 20),
                      _MenuCard(
                        label: 'Leaderboard',
                        subtitle: 'View current and former tournament leaderboards.',
                        onTap: () {
                          Navigator.of(context).push(
                            MaterialPageRoute(
                              builder: (_) => const TournamentResultsPage(),
                            ),
                          );
                        },
                      ),
                      const SizedBox(height: 14),
                      const _MenuCard(
                        label: 'Round History',
                        subtitle: 'Review uploaded scorecards and round history.',
                      ),
                      const SizedBox(height: 14),
                      const _MenuCard(
                        label: 'Upload',
                        subtitle: 'Scan and upload scorecards as players finish each day.',
                      ),
                      const SizedBox(height: 14),
                      const _MenuCard(
                        label: 'Admin',
                        subtitle: 'Create, adjust and manage tournament paramaters.',
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 28),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  _FooterLink(label: 'How It Works', onTap: () {}),
                  _FooterLink(label: 'Help & Support', onTap: () {}),
                ],
              ),
              const SizedBox(height: 8),
            ],
          ),
        ),
      ),
    );
  }
}

class TournamentResultsPage extends StatelessWidget {
  const TournamentResultsPage({super.key});

  static const List<_LeaderboardRow> _activeTournamentLeaderboard = [
    _LeaderboardRow(name: 'Ava Mitchell', totalToPar: '-9', r1: '68', r2: '70', r3: '69', r4: '--'),
    _LeaderboardRow(name: 'Liam Carter', totalToPar: '-7', r1: '69', r2: '71', r3: '69', r4: '--'),
    _LeaderboardRow(name: 'Noah Bennett', totalToPar: '-6', r1: '70', r2: '69', r3: '71', r4: '--'),
    _LeaderboardRow(name: 'Sophia Reed', totalToPar: '-5', r1: '71', r2: '70', r3: '70', r4: '--'),
    _LeaderboardRow(name: 'Mason Turner', totalToPar: '-4', r1: '72', r2: '69', r3: '71', r4: '--'),
    _LeaderboardRow(name: 'Isabella Ward', totalToPar: '-3', r1: '71', r2: '72', r3: '70', r4: '--'),
    _LeaderboardRow(name: 'Ethan Brooks', totalToPar: '-2', r1: '72', r2: '71', r3: '71', r4: '--'),
    _LeaderboardRow(name: 'Olivia Hayes', totalToPar: '-1', r1: '73', r2: '70', r3: '72', r4: '--'),
    _LeaderboardRow(name: 'Lucas Perry', totalToPar: 'E', r1: '72', r2: '72', r3: '72', r4: '--'),
    _LeaderboardRow(name: 'Charlotte Price', totalToPar: '+1', r1: '73', r2: '72', r3: '72', r4: '--'),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1B2A),
      appBar: AppBar(
        backgroundColor: const Color(0xFF142234),
        foregroundColor: Colors.white,
        title: const Text('Tournament Results'),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _ActiveTournamentSection(leaderboardRows: _activeTournamentLeaderboard),
              const SizedBox(height: 20),
              const _PreviousTournamentsSection(
                tournaments: [
                  'Winter Classic - Final Results',
                  'Autumn Member Cup - Final Results',
                  'Summer Pro-Am - Final Results',
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ActiveTournamentSection extends StatelessWidget {
  final List<_LeaderboardRow> leaderboardRows;

  const _ActiveTournamentSection({required this.leaderboardRows});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF142234),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFF1F3A56)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Active Tournaments',
            style: TextStyle(
              color: Color(0xFF4FC3F7),
              fontSize: 18,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 10),
          const Text(
            'Spring Invitational - Pebble Ridge GC',
            style: TextStyle(
              color: Colors.white,
              fontSize: 15,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 14),
          _LeaderboardHeaderRow(),
          const SizedBox(height: 8),
          ...leaderboardRows.take(10).map(_LeaderboardDataRow.new),
          const SizedBox(height: 8),
          Align(
            alignment: Alignment.center,
            child: TextButton(
              onPressed: () {
                ScaffoldMessenger.of(context)
                  ..hideCurrentSnackBar()
                  ..showSnackBar(
                    const SnackBar(content: Text('Full leaderboard coming soon.')),
                  );
              },
              child: const Text(
                'See Full Leaderboard',
                style: TextStyle(
                  color: Color(0xFF4FC3F7),
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _LeaderboardHeaderRow extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return const Row(
      children: [
        Expanded(flex: 4, child: _LeaderboardCell(text: 'Name', isHeader: true)),
        Expanded(flex: 2, child: _LeaderboardCell(text: 'Total', isHeader: true)),
        Expanded(child: _LeaderboardCell(text: 'R1', isHeader: true)),
        Expanded(child: _LeaderboardCell(text: 'R2', isHeader: true)),
        Expanded(child: _LeaderboardCell(text: 'R3', isHeader: true)),
        Expanded(child: _LeaderboardCell(text: 'R4', isHeader: true)),
      ],
    );
  }
}

class _LeaderboardDataRow extends StatelessWidget {
  final _LeaderboardRow row;

  const _LeaderboardDataRow(this.row);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Expanded(flex: 4, child: _LeaderboardCell(text: row.name)),
          Expanded(flex: 2, child: _LeaderboardCell(text: row.totalToPar)),
          Expanded(child: _LeaderboardCell(text: row.r1)),
          Expanded(child: _LeaderboardCell(text: row.r2)),
          Expanded(child: _LeaderboardCell(text: row.r3)),
          Expanded(child: _LeaderboardCell(text: row.r4)),
        ],
      ),
    );
  }
}

class _LeaderboardCell extends StatelessWidget {
  final String text;
  final bool isHeader;

  const _LeaderboardCell({required this.text, this.isHeader = false});

  @override
  Widget build(BuildContext context) {
    return Text(
      text,
      overflow: TextOverflow.ellipsis,
      style: TextStyle(
        color: isHeader ? const Color(0xFF9FB3C8) : const Color(0xFFD6E3F0),
        fontSize: isHeader ? 12 : 13,
        fontWeight: isHeader ? FontWeight.w600 : FontWeight.w500,
      ),
    );
  }
}

class _LeaderboardRow {
  final String name;
  final String totalToPar;
  final String r1;
  final String r2;
  final String r3;
  final String r4;

  const _LeaderboardRow({
    required this.name,
    required this.totalToPar,
    required this.r1,
    required this.r2,
    required this.r3,
    required this.r4,
  });
}

class _PreviousTournamentsSection extends StatelessWidget {
  final List<String> tournaments;

  const _PreviousTournamentsSection({required this.tournaments});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF142234),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFF1F3A56)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Previous Tournaments',
            style: TextStyle(
              color: Color(0xFF4FC3F7),
              fontSize: 18,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 12),
          ...tournaments.map(
            (tournament) => Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Padding(
                    padding: EdgeInsets.only(top: 5),
                    child: Icon(
                      Icons.circle,
                      color: Color(0xFF9FB3C8),
                      size: 8,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      tournament,
                      style: const TextStyle(
                        color: Color(0xFFD6E3F0),
                        fontSize: 14,
                        height: 1.3,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _DirectorOverviewCard extends StatelessWidget {
  const _DirectorOverviewCard();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF142234),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFF1F3A56)),
      ),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Director Overview',
            style: TextStyle(
              color: Color(0xFF4FC3F7),
              fontSize: 17,
              fontWeight: FontWeight.w600,
            ),
          ),
          SizedBox(height: 14),
          _DirectorInfoRow(label: 'Name', value: 'Jordan Whitaker'),
          SizedBox(height: 8),
          _DirectorInfoRow(label: 'Club', value: 'Pebble Ridge Golf Club'),
          SizedBox(height: 8),
          _DirectorInfoRow(label: 'Association', value: 'Midwest Amateur Golf Association'),
        ],
      ),
    );
  }
}

class _DirectorInfoRow extends StatelessWidget {
  final String label;
  final String value;

  const _DirectorInfoRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return RichText(
      text: TextSpan(
        text: '$label: ',
        style: const TextStyle(
          color: Color(0xFF9FB3C8),
          fontSize: 13,
          fontWeight: FontWeight.w500,
        ),
        children: [
          TextSpan(
            text: value,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

class _MenuCard extends StatelessWidget {
  final String label;
  final String subtitle;
  final VoidCallback? onTap;

  const _MenuCard({required this.label, required this.subtitle, this.onTap});

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 16),
          decoration: BoxDecoration(
            color: const Color(0xFF142234),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: const Color(0xFF1F3A56)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: const TextStyle(
                  color: Color(0xFF4FC3F7),
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.3,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                subtitle,
                style: const TextStyle(
                  color: Color(0xFF9FB3C8),
                  fontSize: 13,
                  height: 1.4,
                ),
              ),
            ],
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
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
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
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
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
