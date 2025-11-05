-- Iris Dataset

---------------------------------------------------------------------------------------
-- Inputs & Validation

type Input = Tensor Real [4]
sepal_length    = 0     -- cm
sepal_width     = 1     -- cm
petal_length    = 2     -- cm
petal_width     = 3     -- cm

-- Min, max
-- minimumInputValues : Input
-- minimumInputValues = [4.3, 2.0, 1.0, 0.1]
-- maximumInputValues : Input
-- maximumInputValues = [7.9, 4.4, 6.9, 2.5]
-- validInputRange : Input -> Bool
-- validInputRange x = forall i .
--   minimumInputValues!i <= x!i <= maximumInputValues!i 

validInputRange : Input -> Bool
validInputRange  x =
  4.3 <= x!sepal_length <= 7.9  and
  2.0 <= x!sepal_width <= 4.4   and
  1.0 <= x!petal_length <= 6.9  and
  0.1 <= x!petal_width <= 2.5

validSizes : Input -> Bool
validSizes x = 
  x!sepal_length >= x!sepal_width and
  x!petal_length >= x!petal_width

validInput : Input -> Bool
validInput x =
  validInputRange x and
  validSizes x

---------------------------------------------------------------------------------------
-- Outputs

type Output = Tensor Real [3]
setosa      = 0
versicolor  = 1
virginica   = 2

---------------------------------------------------------------------------------------
-- NN Model

@network
iris : Input -> Output

outIris : Input -> Output
outIris x = iris(x)

-- Get maximum score for the 3 outputs
isMax : Index 3 -> Input -> Bool
-- isMax i x = forall j . 
--   i != j => 
--     iris x!i > iris x!j
isMax i x = 
    let scores = iris x in
    forall d . 
      d != i => 
        scores!i > scores!d

--------------------------------------------------------------------------------
-- Property 0
-- Basic input validation check

@property
property0 : Bool
property0 = forall x .
    validInput x => 
      iris x!versicolor >= 0

--------------------------------------------------------------------------------
-- Property 1
-- Another check, one of any classes, should always pass
-- !? get Marabou counterexample x: [ 4.3, 2.0, 2.951822, 0.1 ]

@property
property1 : Bool
property1 = forall x .
  validInput x =>
    isMax setosa x or
    isMax versicolor x or
    isMax virginica x

--------------------------------------------------------------------------------
-- Property 2
-- Bounds for Setosa

boundsSetosa : Input -> Bool
boundsSetosa x = 
  x!petal_width <= 1.0 and
  x!petal_length <= 2.0

@property
property2 : Bool
property2 = forall x .
  validInput x and boundsSetosa x => 
    isMax setosa x 

--------------------------------------------------------------------------------
-- Property 3
-- Bounds for Versicolor -

boundsVersicolor : Input -> Bool
boundsVersicolor x = 
  1.0 <= x!petal_width <= 1.25 and
  3.0 <= x!petal_length <= 4.0

@property
property3 : Bool
property3 = forall x .
  validInput x and boundsVersicolor x => 
    isMax versicolor x 

--------------------------------------------------------------------------------
-- Property 4
-- Bounds for Virginica -

boundsVirginica : Input -> Bool
boundsVirginica x = 
  x!petal_width >= 2.0 and
  x!petal_length >= 5.0

@property
property4 : Bool
property4 = forall x .
  validInput x and boundsVirginica x => 
    isMax virginica x 