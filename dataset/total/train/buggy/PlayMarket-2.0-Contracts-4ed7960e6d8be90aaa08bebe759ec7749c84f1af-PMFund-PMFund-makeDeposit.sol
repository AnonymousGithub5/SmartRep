function FunctionDefinition_0 ( address _token ) external ModifierInvocation_0 { assert ( _token != address ( 0 ) ) ; assert ( Identifier_0 [ _token ] [ address ( this ) ] == 0 ) ; Identifier_1 [ _token ] [ address ( this ) ] = Identifier_2 ( _token ) . balanceOf ( address ( this ) ) ; Identifier_3 . push ( Identifier_4 ( { token : _token , decimals : Identifier_5 ( _token ) . MemberAccess_0 ( ) , total : Identifier_6 [ _token ] [ address ( this ) ] } ) ) ; }