import 'package:flutter/material.dart';

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
            'Active Tournament',
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
          const _LeaderboardHeaderRow(),
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
  const _LeaderboardHeaderRow();

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
