// ignore_for_file: avoid_print

import 'dart:async';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_database/firebase_database.dart' as dabase;
import 'package:firebase_messaging/firebase_messaging.dart';

import '../helper/utility.dart';
import '../model/feedModel.dart';
import '../model/notificationModel.dart';
import '../model/user.dart';
import '../resource/push_notification_service.dart';
import '../ui/page/common/locator.dart';
import 'appState.dart';

class NotificationState extends AppState {
  // String fcmToken;
  // FeedModel notificationTweetModel;

  // FcmNotificationModel notification;
  // String notificationSenderId;
  Query? query;
  List<UserModel> userList = [];

  List<NotificationModel>? _notificationList;

  addNotificationList(NotificationModel model) {
    _notificationList ??= <NotificationModel>[];

    if (!_notificationList!.any((element) => element.id == model.id)) {
      _notificationList!.insert(0, model);
    }
  }

  List<NotificationModel>? get notificationList => _notificationList;

  /// [Intitilise firebase notification kDatabase]
  Future<bool> databaseInit(String userId) {
    try {
      if (query != null) {
        query = null;
        _notificationList = null;
      }
      query = kFirestore.collection("notification").where(userId = userId);
      query!.snapshots().listen((event) {
        for (var change in event.docChanges) {
          if (change.type == DocumentChangeType.added) {
            for (var doc in event.docs) {
              _onNotificationAdded(doc);
            }
          }
          if (change.type == DocumentChangeType.modified) {
            for (var doc in event.docs) {
              _onNotificationChanged(doc);
            }
          }
          if (change.type == DocumentChangeType.removed) {
            for (var doc in event.docs) {
              _onNotificationRemoved(doc);
            }
          }
        }
      });
      return Future.value(true);
    } catch (error) {
      cprint(error, errorIn: 'databaseInit');
      return Future.value(false);
    }
  }

  /// get [Notification list] from firebase realtime database
  void getDataFromDatabase(String userId) {
    try {
      if (_notificationList != null) {
        return;
      }
      isBusy = true;
      kFirestore
          .collection('notification')
          .doc(userId)
          .get()
          .then((DocumentSnapshot snapshot) {
        if (snapshot.exists) {
          var map = snapshot.data() as Map<dynamic, dynamic>?;
          if (map != null) {
            map.forEach((tweetKey, value) {
              var map = value as Map<dynamic, dynamic>;
              var model = NotificationModel.fromJson(tweetKey, map);
              addNotificationList(model);
            });
            _notificationList!
                .sort((x, y) => x.timeStamp!.compareTo(y.timeStamp!));
          }
        }
        isBusy = false;
      });
    } catch (error) {
      isBusy = false;
      cprint(error, errorIn: 'getDataFromDatabase');
    }
  }

  /// get `Tweet` present in notification
  Future<FeedModel?> getTweetDetail(String tweetId) async {
    FeedModel _tweetDetail;
    await kFirestore
        .collection('tweet')
        .doc(tweetId)
        .get()
        .then((DocumentSnapshot snapshot) {
      if (snapshot.exists) {
        var map = snapshot.data() as Map<dynamic, dynamic>;
        _tweetDetail = FeedModel.fromJson(map);
        _tweetDetail.key = snapshot.id!;
        return _tweetDetail;
      } else {
        return null;
      }
    });
  }

  /// get user who liked your tweet
  Future<UserModel?> getUserDetail(String userId) async {
    UserModel user;
    if (userList.isNotEmpty && userList.any((x) => x.userId == userId)) {
      return Future.value(userList.firstWhere((x) => x.userId == userId));
    }
    await kFirestore
        .collection('profile')
        .doc(userId)
        .get()
        .then((DocumentSnapshot snapshot) {
      if (snapshot.exists) {
        var map = snapshot.data() as Map<dynamic, dynamic>;
        user = UserModel.fromJson(map);
        user.key = snapshot.id!;
        userList.add(user);
        return user;
      } else {
        return null;
      }
    });
  }

  /// Remove notification if related Tweet is not found or deleted
  void removeNotification(String userId, String tweetkey) async {
    kFirestore.collection('notification').doc(userId + "_" + tweetkey).delete();
  }

  /// Trigger when somneone like your tweet
  void _onNotificationAdded(DocumentSnapshot snapshot) {
    if (snapshot.exists) {
      var map = snapshot.data() as Map<dynamic, dynamic>;
      var model = NotificationModel.fromJson(snapshot.id!, map);

      addNotificationList(model);
      // added notification to list
      print("Notification added");
      notifyListeners();
    }
  }

  /// Trigger when someone changed his like preference
  void _onNotificationChanged(DocumentSnapshot snapshot) {
    if (snapshot.exists) {
      notifyListeners();
      print("Notification changed");
    }
  }

  /// Trigger when someone undo his like on tweet
  void _onNotificationRemoved(DocumentSnapshot snapshot) {
    if (snapshot.exists) {
      var map = snapshot.data() as Map<dynamic, dynamic>;
      var model = NotificationModel.fromJson(snapshot.id!, map);
      // remove notification from list
      _notificationList!.removeWhere((x) => x.tweetKey == model.tweetKey);
      notifyListeners();
      print("Notification Removed");
    }
  }

  /// Initilise push notification services
  void initFirebaseService() {
    if (!getIt.isRegistered<PushNotificationService>()) {
      getIt.registerSingleton<PushNotificationService>(
          PushNotificationService(FirebaseMessaging.instance));
    }
  }

  @override
  void dispose() {
    getIt.unregister<PushNotificationService>();
    super.dispose();
  }
}
