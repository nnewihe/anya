import 'package:flutter_test/flutter_test.dart';

import 'package:anya_tennis/main.dart';

void main() {
  testWidgets('App renders home screen', (WidgetTester tester) async {
    await tester.pumpWidget(const AnyaTennisApp());
    expect(find.text('Choose a match video'), findsOneWidget);
  });
}
