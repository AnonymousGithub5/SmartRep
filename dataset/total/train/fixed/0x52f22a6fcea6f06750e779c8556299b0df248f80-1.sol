function FunctionDefinition_0 ( address _from , address _to , uint256 _value ) internal { if ( Identifier_0 [ _from ] < _value ) revert ( ) ; if ( now < Identifier_1 [ _from ] ) revert ( ) ; if ( _to == address ( this ) ) { Identifier_2 ( _value ) ; } else { int256 VariableDeclaration_0 = ( int256 ) ( Identifier_3 * _value ) ; Identifier_4 [ _from ] -= _value ; Identifier_5 [ _to ] += _value ; require ( Identifier_6 [ _to ] >= _value ) ; Identifier_7 [ _from ] -= Identifier_8 ; Identifier_9 [ _to ] += Identifier_10 ; } Transfer ( _from , _to , _value ) ; }