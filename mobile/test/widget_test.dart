import 'package:flutter_test/flutter_test.dart';

import 'package:rally_predictor/main.dart';

void main() {
  testWidgets('App renders home screen', (WidgetTester tester) async {
    await tester.pumpWidget(const RallyPredictorApp());
    expect(find.text('Rally Predictor'), findsOneWidget);
  });
}
