-- ACAS XU

---------------------------------------------------------------------------------------
-- Inputs & Validation

pi = 3.141592

type Input = Tensor Real [5]
distanceToIntruder  = 0     -- metres
angleToIntruder     = 1     -- radians
intruderHeading     = 2     -- radians
speed               = 3     -- metres/second
intruderSpeed       = 4     -- metres/second

-- Problem Space Inputs. 
-- Not yet normalised, so a validation and normalise function
type UnnormalisedInput = Tensor Real [5]

minimumInputValues : UnnormalisedInput
minimumInputValues = [0, -pi, -pi, 0, 0]
maximumInputValues : UnnormalisedInput
maximumInputValues = [60261.0, pi, pi, 1200.0, 1200.0]

validInput : UnnormalisedInput -> Bool
validInput x = forall i .
  minimumInputValues!i <= x!i <= maximumInputValues!i

-- Input Space. Problem Space is normalised for the NN
meanScalingValues : UnnormalisedInput
meanScalingValues = [19791.091, 0.0, 0.0, 650.0, 600.0] -- TODO: Check where these came from?

normalise : UnnormalisedInput -> Input
normalise x = foreach i .
  (x!i - meanScalingValues!i) / (maximumInputValues!i - minimumInputValues!i)

---------------------------------------------------------------------------------------
-- Outputs

type Output = Tensor Real [5]
clearOfConflict = 0
weakLeft        = 1
weakRight       = 2
strongLeft      = 3
strongRight     = 4

---------------------------------------------------------------------------------------
-- NN Model

@network
acasXu : Input -> Output

-- With normalisation and function to apply the NN
normAcasXu : UnnormalisedInput -> Output
normAcasXu x = acasXu(normalise x)

-- For the NN output vector, advises if i is the minimal score
-- rename to minimalScore as per tutorial
minimalScore : Index 5 -> UnnormalisedInput -> Bool
minimalScore i x = forall j .
    -- i != j => normAcasXu x!i < normAcasXu x!j  -- amended to remove warning
    i != j => 
        normAcasXu x!i <= normAcasXu x!j  

---------------------------------------------------------------------------------------
-- ACAS Xu Property 1: 
-- If the intruder is distant and is significantly slower than the ownship, 
--   then the score of a COC advisory will always be below a certain fixed threshold.

isDistant: UnnormalisedInput -> Bool
isDistant x =
    x!distanceToIntruder >= 55947.691 

isVSlower: UnnormalisedInput -> Bool
isVSlower x =
    x!speed         >= 1145 and 
    x!intruderSpeed <= 60

scaledOutputCOC : Real -> Real
scaledOutputCOC x = (x - 7.518884) / 373.94992

@property
property1 : Bool
property1 = forall x .
    validInput x and isDistant x and isVSlower x =>
        normAcasXu x!clearOfConflict <= scaledOutputCOC 1500


--------------------------------------------------------------------------------
-- ACAS Xu Property 3
-- If the intruder is directly ahead and is moving towards the ownship, 
--   then the score for COC will not be minimal.

-- Input constraints: 1500 ≤ρ≤1800,−0.06 ≤θ≤0.06, ψ≥3.10, vown ≥980, vint ≥960.
-- Desired output property: the score for COC is not the minimal score

directlyAhead : UnnormalisedInput -> Bool
directlyAhead x =
  1500  <= x!distanceToIntruder <= 1800 and
  -0.06 <= x!angleToIntruder    <= 0.06

movingTowards : UnnormalisedInput -> Bool
movingTowards x =
  x!intruderHeading >= 3.10  and
  x!speed           >= 980   and
  x!intruderSpeed   >= 960

@property
property3 : Bool
property3 = forall x .
  validInput x and directlyAhead x and movingTowards x =>
    not (minimalScore clearOfConflict x)

--------------------------------------------------------------------------------
-- ACAS Xu Property 5
-- If the intruder is near and approaching from the left, 
--  then the network advises “strong right”.

-- Input constraints: 250 ≤ ρ ≤ 400, 0.2 ≤ θ ≤ 0.4,−3.141592 ≤ ψ ≤ 3.141592 + 0.005, 100 ≤vown ≤400, 0 ≤vint ≤400.
-- Desired output property: the score for “strong right” is the minimal score.

isNear: UnnormalisedInput -> Bool
isNear x =
    250 <= x!distanceToIntruder <= 400

isApproachFromLeft: UnnormalisedInput -> Bool
isApproachFromLeft x =
  0.2 <= x!angleToIntruder <= 0.4 and
  -pi <= x!intruderHeading <= pi + 0.005 and
  100 <= x!speed <=400 and
  0 <= x!intruderSpeed <= 400

@property
property5 : Bool
property5 = forall x .
  validInput x and isNear x and isApproachFromLeft x =>
    minimalScore strongRight x

--------------------------------------------------------------------------------
-- ACAS Xu Property 8
-- For a large vertical separation and a previous “weak left” advisory, 
--  then the network will either output COC or continue advising “weak left”.

-- Input constraints: 0 ≤ρ≤60760,−3.141592 ≤θ≤−0.75·3.141592,−0.1 ≤ ψ≤0.1, 600 ≤vown ≤1200, 600 ≤vint ≤1200.
-- Desired output property: the score for “weak left” is minimal or the score for COC is minimal.