import 'package:flutter/material.dart';
import 'package:flutter_akolo_finbot/helper/enum.dart';
import 'package:flutter_akolo_finbot/model/feedModel.dart';
import 'package:flutter_akolo_finbot/state/authState.dart';
import 'package:flutter_akolo_finbot/state/feedState.dart';
import 'package:flutter_akolo_finbot/ui/theme/theme.dart';
import 'package:flutter_akolo_finbot/widgets/chart.dart';
import 'package:flutter_akolo_finbot/widgets/customWidgets.dart';
import 'package:flutter_akolo_finbot/widgets/newWidget/customLoader.dart';
import 'package:flutter_akolo_finbot/widgets/newWidget/emptyList.dart';
import 'package:flutter_akolo_finbot/widgets/tweet/tweet.dart';
import 'package:flutter_akolo_finbot/widgets/tweet/widgets/tweetBottomSheet.dart';
import 'package:provider/provider.dart';

class SummaryFeedPage extends StatelessWidget {
  const SummaryFeedPage(
      {Key? key, required this.scaffoldKey, this.refreshIndicatorKey})
      : super(key: key);

  final GlobalKey<ScaffoldState> scaffoldKey;

  final GlobalKey<RefreshIndicatorState>? refreshIndicatorKey;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: TwitterColor.mystic,
      body: SafeArea(
        child: SizedBox(
          height: context.height,
          width: context.width,
          child: const ChartWidget(),
        ),
      ),
    );
  }
}
