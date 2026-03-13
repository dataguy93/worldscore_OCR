import 'package:flutter/material.dart';

import 'leaderboard_dummy_data.dart';

class FullLeaderboardPage extends StatefulWidget {
  const FullLeaderboardPage({super.key});

  @override
  State<FullLeaderboardPage> createState() => _FullLeaderboardPageState();
}

class _FullLeaderboardPageState extends State<FullLeaderboardPage> {
  int _selectedDivision = 1;

  List<TournamentRoundEntry> get _divisionEntries {
    final entries = activeTournamentDummyData
        .where((entry) =>
            entry.tournamentName == activeTournamentName &&
            entry.division == _selectedDivision &&
            entry.roundScores.length >= 3)
        .toList()
      ..sort((a, b) {
        final totalA = a.roundScores.take(3).reduce((x, y) => x + y);
        final totalB = b.roundScores.take(3).reduce((x, y) => x + y);
        final byTotal = totalA.compareTo(totalB);
        if (byTotal != 0) return byTotal;
        return a.playerName.compareTo(b.playerName);
      });

    return entries;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1B2A),
      appBar: AppBar(
        backgroundColor: const Color(0xFF142234),
        foregroundColor: Colors.white,
        title: const Text('Full Leaderboard'),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                activeTournamentName,
                style: TextStyle(
                  color: Color(0xFFD6E3F0),
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  const Text(
                    'Division:',
                    style: TextStyle(
                      color: Color(0xFF9FB3C8),
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    decoration: BoxDecoration(
                      color: const Color(0xFF142234),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: const Color(0xFF1F3A56)),
                    ),
                    child: DropdownButton<int>(
                      value: _selectedDivision,
                      dropdownColor: const Color(0xFF142234),
                      underline: const SizedBox.shrink(),
                      style: const TextStyle(color: Colors.white),
                      iconEnabledColor: const Color(0xFF4FC3F7),
                      items: List.generate(
                        5,
                        (index) => DropdownMenuItem<int>(
                          value: index + 1,
                          child: Text('Division ${index + 1}'),
                        ),
                      ),
                      onChanged: (value) {
                        if (value == null) return;
                        setState(() {
                          _selectedDivision = value;
                        });
                      },
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Expanded(
                child: SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: SizedBox(
                    width: 560,
                    child: Column(
                      children: [
                        const _LeaderboardHeader(),
                        const SizedBox(height: 8),
                        Expanded(
                          child: ListView.separated(
                            itemCount: _divisionEntries.length,
                            separatorBuilder: (_, __) => const SizedBox(height: 8),
                            itemBuilder: (context, index) {
                              final entry = _divisionEntries[index];
                              return _LeaderboardEntryRow(rank: index + 1, entry: entry);
                            },
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _LeaderboardHeader extends StatelessWidget {
  const _LeaderboardHeader();

  @override
  Widget build(BuildContext context) {
    return const Row(
      children: [
        _Cell(text: '#', width: 36, isHeader: true),
        _Cell(text: 'Player', width: 170, isHeader: true),
        _Cell(text: 'HCP', width: 60, isHeader: true),
        _Cell(text: 'R1', width: 50, isHeader: true),
        _Cell(text: 'R2', width: 50, isHeader: true),
        _Cell(text: 'R3', width: 50, isHeader: true),
        _Cell(text: 'Net', width: 70, isHeader: true),
        _Cell(text: 'Total', width: 70, isHeader: true),
      ],
    );
  }
}

class _LeaderboardEntryRow extends StatelessWidget {
  final int rank;
  final TournamentRoundEntry entry;

  const _LeaderboardEntryRow({required this.rank, required this.entry});

  @override
  Widget build(BuildContext context) {
    final totalStrokes = entry.roundScores.take(3).reduce((a, b) => a + b);
    final totalToPar = totalStrokes - (72 * 3);
    final netTotalToPar = entry.roundScores
        .take(3)
        .map((roundScore) => (roundScore - 72) - entry.handicap)
        .reduce((a, b) => a + b);

    return Row(
      children: [
        _Cell(text: '$rank', width: 36),
        _Cell(text: entry.playerName, width: 170),
        _Cell(text: entry.handicap.toStringAsFixed(1), width: 60),
        _Cell(text: '${entry.roundScores[0]}', width: 50),
        _Cell(text: '${entry.roundScores[1]}', width: 50),
        _Cell(text: '${entry.roundScores[2]}', width: 50),
        _Cell(text: _formatNetToPar(netTotalToPar), width: 70),
        _Cell(text: _formatToPar(totalToPar), width: 70),
      ],
    );
  }
}

String _formatToPar(int value) {
  if (value == 0) return 'E';
  if (value > 0) return '+$value';
  return '$value';
}

String _formatNetToPar(double value) {
  if (value.abs() < 0.05) return 'E';
  final rounded = value.toStringAsFixed(1);
  return value > 0 ? '+$rounded' : rounded;
}

class _Cell extends StatelessWidget {
  final String text;
  final double width;
  final bool isHeader;

  const _Cell({
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
