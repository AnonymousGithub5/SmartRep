function transfer ( address to , uint256 value ) external returns ( bool ) { require ( Identifier_0 . MemberAccess_0 ( msg . sender , to ) ) ; if ( Identifier_1 != Identifier_2 ( 0 ) ) { require ( Identifier_3 . MemberAccess_1 ( msg . sender , to , value ) , stringLiteral_0 ) ; } else { Identifier_4 ( msg . sender , to , value ) ; } return true ; }