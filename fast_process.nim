import std/json
import std/lenientops
import std/terminal
import times, strutils
import arraymancer
import argParse
import suru

var parser = newParser:
    help("This script generates event-based and alert-based metrics used for downstream calculation of metrics such as precision, recall, sensitivity, specificity, etc.")
    option("-i", "--inputfile", help="The location of the input trajectory data")
    option("-o", "--outputfile", help="The location of the output data", default=some("metrics_output.json"))
    option("-n", "--numthresholds", help="The number of thresholds to consider. Creates an equally-spaced array from 0 to 1.", default=some("100"))
    option("-d", "--detectionwindow", help="Size of detection window")
    option("-s", "--snoozewindow", help="Size of the snoozing window")
    
var inputfile: string
var outputfile: string
var granularity: int
var detectionwindow: int
var snoozewindow: float

try:
    let opts = parser.parse()
    if opts.inputfile == "":
        raise newException(UsageError, "==== Input File Not found! ====")
    elif opts.detectionwindow == "":
        raise newException(UsageError, "==== Detection Window not found! ====")
    elif opts.snoozewindow == "":
        raise newException(USageError, "==== Snoozing Window not found! ====")
    else:
        echo "Running with the following arguments: \n"
        echo "-------------------------------------"
        stdout.styledWriteLine(fgDefault, "Input file path: ", fgGreen, opts.inputfile)
        stdout.styledWriteLine(fgDefault, "Output file path: ", fgGreen, opts.outputfile)
        stdout.styledWriteLine(fgDefault, "Number of thresholds: ", fgRed, opts.numthresholds)
        stdout.styledWriteLine(fgDefault, "Size of detection window: ", fgRed, opts.detectionwindow)
        stdout.styledWriteLine(fgDefault, "Size of snoozing window: ", fgRed, opts.snoozewindow)
      
        inputfile = opts.inputfile
        outputfile = opts.outputfile
        granularity = parseInt(opts.numthresholds)
        detectionwindow = parseInt(opts.detectionwindow)
        snoozewindow = parseFloat(opts.snoozewindow)

except ShortCircuit:
    echo parser.help
    quit(1)
except UsageError:
    echo parser.help
    stderr.writeLine getCurrentExceptionMsg()
    quit(1)


let THRESHOLDS=linspace(0, 1, granularity)

let trajectoryDataStr = readFile(inputfile)
# Simple benchmarking

type Trajectory = object
    y_true: seq[float]
    y_pred: seq[float]

# Deserialize into Trajectory types
let trajectoryData = parseJson(trajectoryDataStr)
var trajectories: seq[Trajectory]
for res in trajectoryData:
    trajectories.add(res.to(Trajectory))

proc get_valid_times(t: seq[int], snoozing_window: float = 2): seq[int] = 
    var result: seq[int]
    for index, time in t.pairs:
        if index == 0:
            result.add(time)
        else:
            if time - result[len(result)-1] > snoozing_window:
                result.add(time)
    return result

proc get_indices_where_gt_threshold(preds: seq[float], threshold: float): seq[int] =
    var result: seq[int]
    for index, pred in preds.pairs:
        if pred > threshold:
            result.add(index)
    return result

proc get_number_between(times: seq[int], lowerbound: int, upperbound: int): int =
    var result: int
    for t in times:
        if t >= lowerbound and t < upperbound:
            result += 1
    return result

# Serialize the result to json
var output: seq[JsonNode]
let time = epochTime()

var tp_event = 0
var tp_enc = 0
var fp_event = 0
var fp_enc = 0
var fn_enc = 0
var tn_enc = 0

var bar: SuruBar = initSuruBar()
bar[0].total = THRESHOLDS.shape[0] # number of iterations
bar.setup()

for thresh in THRESHOLDS:
    for traj in trajectories:
        var pred_times = get_indices_where_gt_threshold(traj.y_pred, thresh)
        var valid_times = get_valid_times(pred_times, snoozing_window=snoozewindow)

        if traj.y_true[len(traj.y_true)-1] == 1:
            if len(valid_times) == 0:
                fn_enc += 1
            else:
                # Convert to a Tensor for tensorops
                var yPredTensor = traj.y_true.toTensor()
                var earliestOnset = yPredTensor.nonzero()[0,0]
                var earliestPt = max(0, earliest_onset - detectionwindow)

                var numTP = get_number_between(valid_times, earliestPt, earliestOnset) 
                tp_event += numTP

                if numTP == 0:
                    fn_enc += 1
                else:
                    tp_enc += 1        
                var numFP = get_number_between(valid_times, -1, earliestPt)
                fp_enc += numFP

                if numFP > 0:
                    fp_enc += 1
        else:
            if len(valid_times) > 0:
                fp_event += len(valid_times)
                fp_enc += 1
            else:
                tn_enc += 1

    output.add( %* {  "thresh" : thresh,
                    "tp_event" : tp_event,
                      "tp_enc" : tp_enc,
                    "fp_event" : fp_event,
                      "fp_enc" : fp_enc,
                      "fn_enc" : fn_enc,
                      "tn_enc" : tn_enc})
    inc bar
    bar.update(50_000_000)
bar.finish()

let endTime = epochTime() - time
let elapsedTime = endTime.formatFloat(format=ffDecimal, precision=3)
echo "Elapsed time: ", elapsedTime, "s"

writeFile(outputfile, content= pretty(%output))

