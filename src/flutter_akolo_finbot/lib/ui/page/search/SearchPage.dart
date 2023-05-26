import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_akolo_finbot/helper/utility.dart';
import 'package:flutter_akolo_finbot/model/user.dart';
import 'package:flutter_akolo_finbot/model/feedModel.dart';
//import 'package:flutter_akolo_finbot/state/searchState.dart';
import 'package:flutter_akolo_finbot/state/searchTweetState.dart';
import 'package:flutter_akolo_finbot/ui/page/profile/profilePage.dart';
import 'package:flutter_akolo_finbot/ui/page/profile/widgets/circular_image.dart';
import 'package:flutter_akolo_finbot/ui/theme/theme.dart';
import 'package:flutter_akolo_finbot/widgets/customAppBar.dart';
import 'package:flutter_akolo_finbot/widgets/customWidgets.dart';
import 'package:flutter_akolo_finbot/widgets/newWidget/title_text.dart';
import 'package:provider/provider.dart';

class SearchPage extends StatefulWidget {
  const SearchPage({Key? key, this.scaffoldKey}) : super(key: key);

  final GlobalKey<ScaffoldState>? scaffoldKey;

  @override
  State<StatefulWidget> createState() => _SearchPageState();
}

class _SearchPageState extends State<SearchPage> {
  @override
  void initState() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final state = Provider.of<SearchTweetState>(context, listen: false);
      state.resetFilterList();
    });
    super.initState();
  }

  void onSettingIconPressed() {
    Navigator.pushNamed(context, '/TrendsPage');
  }

  @override
  Widget build(BuildContext context) {
    final state = Provider.of<SearchTweetState>(context);
    final list = state.userlist;
    return Scaffold(
      appBar: CustomAppBar(
        scaffoldKey: widget.scaffoldKey,
        icon: AppIcon.settings,
        onActionPressed: onSettingIconPressed,
        onSearchChanged: (text) {
          Utility.logEvent("searching: $text");
          //state.filterByUsername(text);
          state.filterBySymbol(text);
        },
      ),
      body: RefreshIndicator(
        onRefresh: () async {
          state.getDataFromDatabase();
          return Future.value();
        },
        child: ListView.separated(
          addAutomaticKeepAlives: false,
          physics: const BouncingScrollPhysics(),
          itemBuilder: (context, index) => _UserTile(user: list![index]),
          separatorBuilder: (_, index) => const Divider(
            height: 0,
          ),
          itemCount: list?.length ?? 0,
        ),
      ),
    );
  }
}

class _UserTile extends StatelessWidget {
  const _UserTile({Key? key, required this.user}) : super(key: key);
  final FeedModel user;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      onTap: () {
        if (kReleaseMode) {
          kAnalytics.logViewSearchResults(searchTerm: user.userId!);
        }
        Navigator.push(context, ProfilePage.getRoute(profileId: user.userId!));
      },
      leading: CircularImage(path: user.imagePath, height: 40),
      title: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Flexible(
            child: TitleText(user.description!,
                fontSize: 16,
                fontWeight: FontWeight.w800,
                overflow: TextOverflow.ellipsis),
          ),
          const SizedBox(width: 3),
          user.isValidTweet!
              ? customIcon(
                  context,
                  icon: AppIcon.blueTick,
                  isTwitterIcon: true,
                  iconColor: AppColor.primary,
                  size: 13,
                  paddingIcon: 3,
                )
              : const SizedBox(width: 0),
        ],
      ),
      subtitle: Text(user.userId!),
    );
  }
}
