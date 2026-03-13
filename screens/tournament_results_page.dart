import 'package:flutter/material.dart';

import 'full_leaderboard_page.dart';
import 'leaderboard_dummy_data.dart';

class TournamentResultsPage extends StatelessWidget {
  const TournamentResultsPage({super.key});

  List<_LeaderboardRow> get _activeTournamentLeaderboard =>
      _buildLeaderboardRows(activeTournamentDummyData);

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
            activeTournamentName,
            style: TextStyle(
              color: Colors.white,
              fontSize: 15,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 14),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const _LeaderboardHeaderRow(),
                const SizedBox(height: 8),
                ...leaderboardRows.take(10).map(_LeaderboardDataRow.new),
              ],
            ),
          ),
          const SizedBox(height: 8),
          Align(
            alignment: Alignment.center,
            child: TextButton(
              onPressed: () {
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (_) => const FullLeaderboardPage(),
                  ),
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
        _LeaderboardCell(text: 'Name', isHeader: true, width: 170),
        _LeaderboardCell(text: 'Total', isHeader: true, width: 70),
        _LeaderboardCell(text: 'R1', isHeader: true, width: 50),
        _LeaderboardCell(text: 'R2', isHeader: true, width: 50),
        _LeaderboardCell(text: 'R3', isHeader: true, width: 50),
        _LeaderboardCell(text: 'R4', isHeader: true, width: 50),
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
          _LeaderboardCell(text: row.name, width: 170),
          _LeaderboardCell(text: row.totalToPar, width: 70),
          _LeaderboardCell(text: row.r1, width: 50),
          _LeaderboardCell(text: row.r2, width: 50),
          _LeaderboardCell(text: row.r3, width: 50),
          _LeaderboardCell(text: row.r4, width: 50),
        ],
      ),
    );
  }
}

class _LeaderboardCell extends StatelessWidget {
  final String text;
  final bool isHeader;
  final double width;

  const _LeaderboardCell({
    required this.text,
    required this.width,
    this.isHeader = false,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: width,
      child: Text(
        text,
        overflow: TextOverflow.ellipsis,
        style: TextStyle(
          color: isHeader ? const Color(0xFF9FB3C8) : const Color(0xFFD6E3F0),
          fontSize: isHeader ? 12 : 13,
          fontWeight: isHeader ? FontWeight.w600 : FontWeight.w500,
        ),
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


List<_LeaderboardRow> _buildLeaderboardRows(List<TournamentRoundEntry> entries) {
  final rows = entries
      .where((entry) =>
          entry.tournamentName == activeTournamentName &&
          entry.roundScores.length >= 3)
      .map((entry) {
    final totalStrokes = entry.roundScores.take(3).reduce((a, b) => a + b);
    final toPar = totalStrokes - (72 * 3);

    String formatToPar(int value) {
      if (value == 0) return 'E';
      if (value > 0) return '+$value';
      return '$value';
    }

    return _LeaderboardRow(
      name: entry.playerName,
      totalToPar: formatToPar(toPar),
      r1: entry.roundScores[0].toString(),
      r2: entry.roundScores[1].toString(),
      r3: entry.roundScores[2].toString(),
      r4: '--',
    );
  }).toList()
    ..sort((a, b) {
      int scoreValue(String value) {
        if (value == 'E') return 0;
        return int.parse(value);
      }

      final byTotal = scoreValue(a.totalToPar).compareTo(scoreValue(b.totalToPar));
      if (byTotal != 0) return byTotal;
      return a.name.compareTo(b.name);
    });

  return rows;
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
