function FunctionDefinition_0 ( address _from , address _to , uint256 _value ) internal { if ( Identifier_0 [ _from ] < _value ) revert ( ) ; if ( _to == address ( this ) ) { Identifier_1 ( _value ) ; } else { int256 VariableDeclaration_0 = ( int256 ) ( Identifier_2 * _value ) ; Identifier_3 [ _from ] -= _value ; Identifier_4 [ _to ] += _value ; Identifier_5 [ _from ] -= Identifier_6 ; Identifier_7 [ _to ] += Identifier_8 ; } Transfer ( _from , _to , _value ) ; }