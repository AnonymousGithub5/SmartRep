function FunctionDefinition_0 ( ERC20 _token ) public view returns ( uint256 ) { uint256 VariableDeclaration_0 = _token . balanceOf ( this ) ; uint256 VariableDeclaration_1 = Identifier_0 . add ( Identifier_1 [ _token ] ) ; if ( block . timestamp < Identifier_2 ) { return 0 ; } else if ( block . timestamp >= Identifier_3 . add ( Identifier_4 ) || Identifier_5 [ _token ] ) { return Identifier_6 ; } else { return Identifier_7 . mul ( block . timestamp . sub ( Identifier_8 ) ) . div ( Identifier_9 ) ; } }