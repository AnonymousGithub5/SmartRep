function FunctionDefinition_0 ( ERC20 calldata Parameter_0 ) external { require ( address ( Identifier_0 ) != address ( 0 ) , stringLiteral_0 ) ; for ( uint16 i = 0 ; i < _tokens . length ; i ++ ) { ( uint256 expectedRate , ) = Identifier_1 . MemberAccess_0 ( _tokens [ i ] , ERC20 ( Identifier_2 ) , NumberLiteral_0 ) ; Identifier_3 [ address ( _tokens [ i ] ) ] = Identifier_4 ; } }