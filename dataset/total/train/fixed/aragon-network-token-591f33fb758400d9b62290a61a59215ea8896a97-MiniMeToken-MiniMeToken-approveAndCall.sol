function FunctionDefinition_0 ( address _spender , uint256 _amount , bytes Parameter_0 ) returns ( bool success ) { allowed [ msg . sender ] [ _spender ] = _amount ; Identifier_0 ( msg . sender , _spender , _amount ) ; if ( ! _spender . call ( bytes4 ( bytes32 ( Identifier_1 ( stringLiteral_0 ) ) ) , msg . sender , _amount , this , uint256 ( ElementaryTypeName_0 ( NumberLiteral_0 ) ) , uint256 ( Identifier_2 . length ) , Identifier_3 ( Identifier_4 ) ) ) { throw ; } return true ; }