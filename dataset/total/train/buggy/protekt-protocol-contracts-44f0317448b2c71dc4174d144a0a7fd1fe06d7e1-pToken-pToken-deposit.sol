function deposit ( uint256 _amount ) public { uint256 VariableDeclaration_0 = balance ( ) ; uint256 VariableDeclaration_1 = Identifier_0 . balanceOf ( address ( this ) ) ; Identifier_1 . safeTransferFrom ( msg . sender , address ( this ) , _amount ) ; uint256 VariableDeclaration_2 = Identifier_2 . balanceOf ( address ( this ) ) ; _amount = Identifier_3 . sub ( Identifier_4 ) ; uint256 VariableDeclaration_3 = 0 ; if ( totalSupply ( ) == 0 ) { Identifier_5 = _amount ; } else { Identifier_6 = ( _amount . mul ( totalSupply ( ) ) ) . div ( _pool ) ; } _mint ( msg . sender , Identifier_7 ) ; }