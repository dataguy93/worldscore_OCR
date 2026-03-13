class TournamentRoundEntry {
  final DateTime roundDate;
  final String playerName;
  final String playerId;
  final String tournamentName;
  final List<int> roundScores;
  final double handicap;
  final int division;

  const TournamentRoundEntry({
    required this.roundDate,
    required this.playerName,
    required this.playerId,
    required this.tournamentName,
    required this.roundScores,
    required this.handicap,
    required this.division,
  });
}

const String activeTournamentName = 'Club el Rodeo Member Guest 2026';

final List<TournamentRoundEntry> activeTournamentDummyData = List.generate(60, (index) {
  final names = <String>[
    'Ava Mitchell', 'Liam Carter', 'Noah Bennett', 'Sophia Reed', 'Mason Turner',
    'Isabella Ward', 'Ethan Brooks', 'Olivia Hayes', 'Lucas Perry', 'Charlotte Price',
    'Benjamin Scott', 'Amelia Cooper', 'Henry Flores', 'Harper Rivera', 'Jackson Sanders',
    'Ella Bryant', 'Sebastian Ross', 'Grace Coleman', 'Daniel Jenkins', 'Chloe Powell',
    'Levi Hughes', 'Victoria Kelly', 'Wyatt Peterson', 'Nora Gray', 'Julian Simmons',
    'Lily Foster', 'David Butler', 'Zoe Barnes', 'Matthew Russell', 'Aria Henderson',
    'Samuel Long', 'Scarlett Patterson', 'Owen Griffin', 'Layla Hughes', 'Carter Hayes',
    'Hannah Stone', 'Ryan Price', 'Stella Howard', 'Nathan Woods', 'Paisley Bryant',
    'Caleb Greene', 'Riley Bell', 'Jonathan Ward', 'Aubrey Cox', 'Thomas Perry',
    'Addison Fisher', 'Andrew Murphy', 'Violet Richardson', 'Christopher James', 'Mila Brooks',
    'Isaac Watson', 'Aurora Reed', 'Gabriel Lee', 'Savannah Cox', 'Joshua Bennett',
    'Penelope Torres', 'Elijah Hughes', 'Madison Kelly', 'Logan Ramirez', 'Camila Ross',
  ];

  final handicap = (((index * 1.25) % 24) + 1.0);
  final division = _divisionFromHandicap(handicap);
  final baseScore = 69 + (handicap / 5).floor();
  final roundScores = <int>[
    baseScore + (index % 4),
    baseScore + ((index + 1) % 4),
    baseScore + ((index + 2) % 4),
  ];

  return TournamentRoundEntry(
    roundDate: DateTime(2026, 5, 14 + (index % 3)),
    playerName: names[index],
    playerId: 'wsai-user-${(index + 1).toString().padLeft(3, '0')}',
    tournamentName: activeTournamentName,
    roundScores: roundScores,
    handicap: double.parse(handicap.toStringAsFixed(1)),
    division: division,
  );
});

int _divisionFromHandicap(double handicap) {
  if (handicap <= 4.9) return 1;
  if (handicap <= 9.9) return 2;
  if (handicap <= 14.9) return 3;
  if (handicap <= 19.9) return 4;
  return 5;
}
