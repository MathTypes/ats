import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter_akolo_finbot/helper/enum.dart';
import 'package:flutter_akolo_finbot/helper/utility.dart';
import 'package:flutter_akolo_finbot/model/feedModel.dart';
import 'package:flutter_akolo_finbot/model/user.dart';
import 'appState.dart';

class SearchTweetState extends AppState {
  bool isBusy = false;
  SortUser sortBy = SortUser.MaxFollower;
  List<FeedModel>? _tweetFilterList;
  List<FeedModel>? _tweetlist;

  List<FeedModel>? get userlist {
    if (_tweetFilterList == null) {
      return null;
    } else {
      return List.from(_tweetFilterList!);
    }
  }

  /// get [UserModel list] from firebase realtime Database
  void getDataFromDatabase() {
    try {
      isBusy = true;
      kFirestore
          .collection('recent_tweet_by_user')
          .limit(1000)
          .snapshots()
          .listen(
        (QuerySnapshot snapshot) {
          _tweetlist = <FeedModel>[];
          _tweetFilterList = <FeedModel>[];
          var map = snapshot.docs;
          map.forEach((value) {
            Utility.logEvent(value.toString());
            var model = FeedModel.fromJson(value.data() as Map);
            model.key = value.id;
            _tweetlist!.add(model);
            _tweetFilterList!.add(model);
          });
          _tweetFilterList!
              .sort((x, y) => y.createdAt!.compareTo(x.createdAt!));
          notifyListeners();
          isBusy = false;
        },
      );
    } catch (error) {
      isBusy = false;
      cprint(error, errorIn: 'getDataFromDatabase');
    }
  }

  /// It will reset filter list
  /// If user has use search filter and change screen and came back to search screen It will reset user list.
  /// This function call when search page open.
  void resetFilterList() {
    if (_tweetlist != null && _tweetlist!.length != _tweetFilterList!.length) {
      _tweetFilterList = List.from(_tweetlist!);
      _tweetFilterList!.sort((x, y) => y.createdAt!.compareTo(x.createdAt!));
      // notifyListeners();
    }
  }

  /// This function call when search fiels text change.
  /// UserModel list on  search field get filter by `name` string
  void filterBySymbol(String? name) {
    if (name != null &&
        name.isEmpty &&
        _tweetlist != null &&
        _tweetlist!.length != _tweetFilterList!.length) {
      _tweetFilterList = List.from(_tweetlist!);
    }
    // return if userList is empty or null
    if (_tweetlist == null && _tweetlist!.isEmpty) {
      cprint("User list is empty");
      return;
    }
    // sortBy userlist on the basis of username
    else if (name != null) {
      _tweetFilterList = _tweetlist!
          .where((x) =>
              x.description != null &&
              x.description!.toLowerCase().contains(name.toLowerCase()))
          .toList();
    }
    notifyListeners();
  }

  /// This function call when search fiels text change.
  /// UserModel list on  search field get filter by `name` string
  void filterByUsername(String? name) {
    if (name != null &&
        name.isEmpty &&
        _tweetlist != null &&
        _tweetlist!.length != _tweetFilterList!.length) {
      _tweetFilterList = List.from(_tweetlist!);
    }
    // return if userList is empty or null
    if (_tweetlist == null && _tweetlist!.isEmpty) {
      cprint("User list is empty");
      return;
    }
    // sortBy userlist on the basis of username
    else if (name != null) {
      _tweetFilterList = _tweetlist!
          .where((x) =>
              x.userId != null &&
              x.userId!.toLowerCase().contains(name.toLowerCase()))
          .toList();
    }
    notifyListeners();
  }

  /// Sort user list on search user page.
  set updateUserSortPrefrence(SortUser val) {
    sortBy = val;
    notifyListeners();
  }

  String get selectedFilter {
    switch (sortBy) {
      case SortUser.Newest:
        _tweetFilterList!.sort((x, y) => y.createdAt!.compareTo(x.createdAt!));
        return "Newest user";

      default:
        return "Unknown";
    }
  }

  /// Return user list relative to provided `userIds`
  /// Method is used on
  List<FeedModel> userList = [];
  List<FeedModel> getuserDetail(List<String> userIds) {
    final list = _tweetlist!.where((x) {
      if (userIds.contains(x.key)) {
        return true;
      } else {
        return false;
      }
    }).toList();
    return list;
  }
}
