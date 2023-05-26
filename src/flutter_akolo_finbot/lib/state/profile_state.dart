import 'dart:async';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_akolo_finbot/helper/enum.dart';
import 'package:flutter_akolo_finbot/helper/utility.dart';
import 'package:flutter_akolo_finbot/model/user.dart';

class ProfileState extends ChangeNotifier {
  ProfileState(this.profileId) {
    databaseInit();
    userId = FirebaseAuth.instance.currentUser!.uid;
    _getloggedInUserProfile(userId);
    _getProfileUser(profileId);
  }

  /// This is the id of user who is logegd into the app.
  late String userId;

  /// Profile data of logged in user.
  late UserModel _userModel;
  UserModel get userModel => _userModel;

  DocumentReference? _profileQuery;
  late StreamSubscription profileSubscription;

  /// This is the id of user whose profile is open.
  final String profileId;

  /// Profile data of user whose profile is open.
  late UserModel _profileUserModel;
  UserModel get profileUserModel => _profileUserModel;

  bool _isBusy = true;
  bool get isbusy => _isBusy;
  set loading(bool value) {
    _isBusy = value;
    notifyListeners();
  }

  databaseInit() {
    try {
      if (_profileQuery == null) {
        _profileQuery = kFirestore.collection("profile").doc(profileId);
        profileSubscription = _profileQuery!.snapshots().listen((event) {
          _onProfileChanged(event);
        });
      }
    } catch (error) {
      cprint(error, errorIn: 'databaseInit');
    }
  }

  bool get isMyProfile => profileId == userId;

  /// Fetch profile of logged in  user
  void _getloggedInUserProfile(String userId) async {
    kFirestore
        .collection("profile")
        .doc(userId)
        .get()
        .then((DocumentSnapshot snapshot) {
      if (snapshot.exists) {
        var map = snapshot.data as Map<dynamic, dynamic>?;
        if (map != null) {
          _userModel = UserModel.fromJson(map);
        }
      }
    });
  }

  /// Fetch profile data of user whoose profile is opened
  void _getProfileUser(String? userProfileId) {
    assert(userProfileId != null);
    try {
      loading = true;
      kFirestore
          .collection("profile")
          .doc(userProfileId!)
          .get()
          .then((DocumentSnapshot snapshot) {
        if (snapshot.exists) {
          var map = snapshot.data as Map;
          // ignore: unnecessary_null_comparison
          if (map != null) {
            _profileUserModel = UserModel.fromJson(map);
            Utility.logEvent('get_profile', parameter: {});
          }
        }
        loading = false;
      });
    } catch (error) {
      loading = false;
      cprint(error, errorIn: 'getProfileUser');
    }
  }

  /// Follow / Unfollow user
  ///
  /// If `removeFollower` is true then remove user from follower list
  ///
  /// If `removeFollower` is false then add user to follower list
  followUser({bool removeFollower = false}) {
    /// `userModel` is user who is logged-in app.
    /// `profileUserModel` is user whoose profile is open in app.
    try {
      if (removeFollower) {
        /// If logged-in user `alredy follow `profile user then
        /// 1.Remove logged-in user from profile user's `follower` list
        /// 2.Remove profile user from logged-in user's `following` list
        profileUserModel.followersList!.remove(userModel.userId);

        /// Remove profile user from logged-in user's following list
        userModel.followingList!.remove(profileUserModel.userId);
        cprint('user removed from following list', event: 'remove_follow');
      } else {
        /// if logged in user is `not following` profile user then
        /// 1.Add logged in user to profile user's `follower` list
        /// 2. Add profile user to logged in user's `following` list
        profileUserModel.followersList ??= [];
        profileUserModel.followersList!.add(userModel.userId!);
        // Adding profile user to logged-in user's following list
        userModel.followingList ??= [];
        addFollowNotification();
        userModel.followingList!.add(profileUserModel.userId!);
      }
      // update profile user's user follower count
      profileUserModel.followers = profileUserModel.followersList!.length;
      // update logged-in user's following count
      userModel.following = userModel.followingList!.length;
      kFirestore
          .collection('profile')
          .doc(profileUserModel.userId!)
          .set({"followersList": profileUserModel.followersList});
      kFirestore
          .collection('profile')
          .doc(userModel.userId!)
          .set({'followingList': userModel.followingList});
      cprint('user added to following list', event: 'add_follow');

      notifyListeners();
    } catch (error) {
      cprint(error, errorIn: 'followUser');
    }
  }

  void addFollowNotification() {
    // Sends notification to user who created tweet
    // UserModel owner can see notification on notification page
    kFirestore.collection('notification').doc(profileId + "_" + userId).set({
      'type': NotificationType.Follow.toString(),
      'createdAt': DateTime.now().toUtc().toString(),
      'data': UserModel(
              displayName: userModel.displayName,
              profilePic: userModel.profilePic,
              isVerified: userModel.isVerified,
              userId: userModel.userId,
              bio: userModel.bio == "Edit profile to update bio"
                  ? ""
                  : userModel.bio,
              userName: userModel.userName)
          .toJson()
    });
  }

  /// Trigger when logged-in user's profile change or updated
  /// Firebase event callback for profile update
  void _onProfileChanged(DocumentSnapshot snapshot) {
    // if (event.snapshot != null) {

    final updatedUser = UserModel.fromJson(snapshot.data as Map);
    if (updatedUser.userId == profileId) {
      _profileUserModel = updatedUser;
    }
    notifyListeners();
    // }
  }

  @override
  void dispose() {
    _profileQuery!.snapshots().drain();
    profileSubscription.cancel();
    // _profileQuery.
    super.dispose();
  }
}
