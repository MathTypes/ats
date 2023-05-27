/// Dart imports
import 'dart:async';
import 'dart:math' as math;

/// Package imports
import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter_akolo_finbot/helper/utility.dart';

/// Chart import
import 'package:syncfusion_flutter_charts/charts.dart';

class ChartWidget extends StatefulWidget {
  const ChartWidget({Key? key, this.table_name}) : super(key: key);

  final String? table_name;
  @override
  State<ChartWidget> createState() => _ChartWidgetState(table_name);
}

/// Private class for storing the chart series data points.
class _ChartData {
  _ChartData(this.time, this.sentiment);

  final int time;
  final num sentiment;
}

class _ChartWidgetState extends State<ChartWidget> {
  _ChartWidgetState(this.tableName) {}

  Timer? timer;
  final String? tableName;
  late List<_ChartData> chartData;
  late int count;
  ChartSeriesController? _chartSeriesController;
  int? _lastTime;

  @override
  void dispose() {
    timer?.cancel();
    chartData.clear();
    _chartSeriesController = null;
    super.dispose();
  }

  @override
  void initState() {
    try {
      Utility.logEvent("init sentiment");
      kFirestore
          .collection('recent_sentiment')
          .orderBy('asof')
          .limitToLast(19)
          .get()
          .then((QuerySnapshot snapshot) {
        //print("getDataFromDatabase_event: $snapshot");
        chartData = <_ChartData>[];
        var map = snapshot.docs;
        map.forEach((value) {
          var data = value.data() as Map;
          Utility.logEvent("sentiment:${data}");
          chartData.add(_ChartData(data["asof"], int(data["sentiment"])));
        });
      });
      _lastTime = chartData!.last.time;
    } catch (error) {
      Utility.logEvent(error.toString());
    }
    super.initState();
    timer =
        Timer.periodic(const Duration(milliseconds: 1000), _updateDataSource);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SfCartesianChart(
        zoomPanBehavior: ZoomPanBehavior(
          enableMouseWheelZooming: true,
          enablePanning: true,
        ),
        plotAreaBorderWidth: 0,
        primaryXAxis: NumericAxis(
          majorGridLines: const MajorGridLines(width: 0),
        ),
        primaryYAxis: NumericAxis(
          axisLine: const AxisLine(width: 0),
          majorTickLines: const MajorTickLines(size: 0),
        ),
        trackballBehavior: TrackballBehavior(
          activationMode: ActivationMode.longPress,
          enable: true,
          lineWidth: 0,
          tooltipDisplayMode: TrackballDisplayMode.nearestPoint,
          tooltipSettings: const InteractiveTooltip(
            arrowLength: 15,
          ),
        ),
        series: <LineSeries<_ChartData, int>>[
          LineSeries<_ChartData, int>(
            onRendererCreated: (ChartSeriesController controller) {
              _chartSeriesController = controller;
            },
            dataSource: chartData,
            color: const Color.fromRGBO(192, 108, 132, 1),
            xValueMapper: (_ChartData value, _) => value.time,
            yValueMapper: (_ChartData value, _) => value.sentiment,
            animationDuration: 0,
          ),
        ],
      ),
    );
  }

  void _updateDataSource(Timer timer) {
    kFirestore
        .collection('recent_sentiment')
        .where('time', isGreaterThan: _lastTime)
        .orderBy('time')
        .limitToLast(19)
        .get()
        .then((QuerySnapshot snapshot) {
      //print("getDataFromDatabase_event: $snapshot");
      chartData = <_ChartData>[];
      var map = snapshot.docs;
      map.forEach((value) {
        var data = value.data() as Map;
        chartData!.add(_ChartData(data["time"], data["sentiment"]));
      });
    });

    while (chartData!.length == 20) {
      chartData.removeAt(0);
      _chartSeriesController?.updateDataSource(
        addedDataIndexes: <int>[chartData.length - 1],
        removedDataIndexes: <int>[0],
      );
    }
  }
}
