import 'dart:io';
import 'package:path/path.dart' as Path;
import 'package:firebase_database/firebase_database.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter_akolo_finbot/helper/enum.dart';
import 'package:flutter_akolo_finbot/helper/utility.dart';
import 'package:flutter_akolo_finbot/model/feedModel.dart';
import 'package:flutter_akolo_finbot/state/appState.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class TweetBaseState extends AppState {
  /// get [Tweet Detail] from firebase realtime kDatabase
  /// If model is null then fetch tweet from firebase
  /// [getPostDetailFromDatabase] is used to set prepare Tweet to display Tweet detail
  /// After getting tweet detail fetch tweet comments from firebase
  Future<FeedModel?> getPostDetailFromDatabase(String postID) async {
    try {
      late FeedModel tweet;
      await kFirestore
          .collection('recent_tweet_by_user')
          .doc(postID)
          .get()
          .then((DocumentSnapshot docSnapshot) {
        if (docSnapshot.exists) {
          tweet =
              FeedModel.fromJson(docSnapshot.data() as Map<dynamic, dynamic>);
          tweet.key = docSnapshot.id;
        }
      });
      return tweet;
    } catch (error) {
      cprint(error, errorIn: 'getPostDetailFromDatabase');
      return null;
    }
  }

  Future<List<FeedModel>?> getTweetsComments(FeedModel post) async {
    late List<FeedModel> _commentList;
    // Check if parent tweet has reply tweets or not
    if (post.replyTweetKeyList != null && post.replyTweetKeyList!.isNotEmpty) {
      // for (String? x in post.replyTweetKeyList!) {
      //   if (x == null) {
      //     return;
      //   }
      // }
      //FIXME
      _commentList = [];
      for (String? replyTweetId in post.replyTweetKeyList!) {
        if (replyTweetId != null) {
          await kFirestore
              .collection('recent_tweet_by_user')
              .doc(replyTweetId)
              .get()
              .then((DocumentSnapshot snapshot) {
            if (snapshot.exists) {
              var commentModel = FeedModel.fromJson(snapshot.data() as Map);
              var key = snapshot.id;
              commentModel.key = key;

              /// add comment tweet to list if tweet is not present in [comment tweet ]list
              /// To reduce delicacy
              if (!_commentList.any((x) => x.key == key)) {
                _commentList.add(commentModel);
              }
            } else {
              if (replyTweetId == post.replyTweetKeyList!.last) {
                /// Sort comment by time
                /// It helps to display newest Tweet first.
                _commentList.sort((x, y) => y.createdAt.compareTo(x.createdAt));
              }
            }
          });
        }
      }
    }
    return _commentList;
  }

  /// [Delete tweet] in Firebase kDatabase
  /// Remove Tweet if present in home page Tweet list
  /// Remove Tweet if present in Tweet detail page or in comment
  bool deleteTweet(
    String tweetId,
    TweetType type,
    /*{String parentkey}*/
  ) {
    try {
      /// Delete tweet if it is in nested tweet detail page
      kFirestore.collection('recent_tweet_by_user').doc(tweetId).delete();
      return true;
    } catch (error) {
      cprint(error, errorIn: 'deleteTweet');
      return false;
    }
  }

  /// [update] tweet
  void updateTweet(FeedModel model) async {
    await kFirestore
        .collection('recent_tweet_by_user')
        .doc(model.key!)
        .set(model.toJson());
  }

  /// Add/Remove like on a Tweet
  /// [postId] is tweet id, [userId] is user's id who like/unlike Tweet
  void addLikeToTweet(FeedModel tweet, String userId) {
    try {
      if (tweet.likeList != null &&
          tweet.likeList!.isNotEmpty &&
          tweet.likeList!.any((id) => id == userId)) {
        // If user wants to undo/remove his like on tweet
        tweet.likeList!.removeWhere((id) => id == userId);
        tweet.likeCount = tweet.likeCount! - 1;
      } else {
        // If user like Tweet
        tweet.likeList ??= [];
        tweet.likeList!.add(userId);
        tweet.likeCount = tweet.likeCount! + 1;
      }
      // update likeList of a tweet
      //kFirestore
      //    .collection('recent_tweet_by_user')
      //    .doc(tweet.key!)
      //    .child('likeList')
      //    .set(tweet.likeList);

      // Sends notification to user who created tweet
      // UserModel owner can see notification on notification page
      kFirestore
          .collection('notification')
          .doc(tweet.userId + "_" + tweet.key!)
          .set({
        'type':
            tweet.likeList!.isEmpty ? null : NotificationType.Like.toString(),
        'updatedAt':
            tweet.likeList!.isEmpty ? null : DateTime.now().toUtc().toString(),
      });
    } catch (error) {
      cprint(error, errorIn: 'addLikeToTweet');
    }
  }

  /// Add new [tweet]
  /// Returns new tweet id
  Future<String?> createPost(FeedModel tweet) async {
    var json = tweet.toJson();
    String? id;
    kFirestore
        .collection('recent_tweet_by_user')
        .add(json)
        .then((value) => (value.id))
        .catchError((error) => "");
    return id;
  }

  /// upload [file] to firebase storage and return its  path url
  Future<String?> uploadFile(File file) async {
    try {
      // isBusy = true;
      notifyListeners();
      var storageReference = FirebaseStorage.instance
          .ref()
          .child("tweetImage")
          .child(Path.basename(DateTime.now().toIso8601String() + file.path));
      await storageReference.putFile(file);

      var url = await storageReference.getDownloadURL();
      return url;
    } catch (error) {
      cprint(error, errorIn: 'uploadFile');
      return null;
    }
  }
}
